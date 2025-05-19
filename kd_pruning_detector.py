import torch
import torch.nn as nn
from ..backbones_3d import build_backbone
from ..dense_heads import build_dense_head
from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils

class CategoryAwarePruning(nn.Module):
    def __init__(self, num_categories, in_channels):
        super().__init__()
        self.num_categories = num_categories
        self.in_channels = in_channels
        

        self.pruning_weights = nn.Parameter(torch.ones(num_categories, in_channels))
        
    def forward(self, features, cls_scores):

        category_importance = torch.softmax(cls_scores, dim=1) 
        
        channel_importance = torch.matmul(category_importance, self.pruning_weights) 
        

        pruned_features = features * channel_importance.unsqueeze(-1).unsqueeze(-1)
        
        return pruned_features, channel_importance

class FeatureProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.projection(x)

class LabelAssignment(nn.Module):
    def __init__(self, iou_threshold=0.6):
        super().__init__()
        self.iou_threshold = iou_threshold
        
    def forward(self, teacher_boxes, student_boxes, gt_boxes):
        """
        Args:
            teacher_boxes: [B, N, 7] predicted boxes from teacher
            student_boxes: [B, M, 7] predicted boxes from student
            gt_boxes: [B, K, 7] ground truth boxes
        Returns:
            teacher_matched_idx: [B, N] indices of matched GT boxes for teacher
            student_matched_idx: [B, M] indices of matched GT boxes for student
        """
        batch_size = teacher_boxes.shape[0]
        teacher_matched_idx = []
        student_matched_idx = []
        
        for b in range(batch_size):

            teacher_iou = model_nms_utils.boxes_iou3d_gpu(
                teacher_boxes[b], gt_boxes[b]
            )

            student_iou = model_nms_utils.boxes_iou3d_gpu(
                student_boxes[b], gt_boxes[b]
            )

            teacher_matched = torch.max(teacher_iou, dim=1)[1]
            student_matched = torch.max(student_iou, dim=1)[1]

            teacher_matched[torch.max(teacher_iou, dim=1)[0] < self.iou_threshold] = -1
            student_matched[torch.max(student_iou, dim=1)[0] < self.iou_threshold] = -1
            
            teacher_matched_idx.append(teacher_matched)
            student_matched_idx.append(student_matched)
            
        return torch.stack(teacher_matched_idx), torch.stack(student_matched_idx)

class SoftTargetGenerator(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, teacher_cls, teacher_reg, matched_idx):

        batch_size = teacher_cls.shape[0]
        soft_cls_targets = []
        soft_reg_targets = []
        
        for b in range(batch_size):

            valid_mask = matched_idx[b] >= 0
            if not valid_mask.any():
                continue

            cls_scores = teacher_cls[b, valid_mask]
            soft_cls = torch.softmax(cls_scores / self.temperature, dim=-1)

            reg_outputs = teacher_reg[b, valid_mask]
            
            soft_cls_targets.append(soft_cls)
            soft_reg_targets.append(reg_outputs)
            
        return soft_cls_targets, soft_reg_targets

class KDPruningDetector(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # Teacher branch
        self.teacher_backbone = build_backbone(model_cfg.TEACHER_BACKBONE)
        self.teacher_head = build_dense_head(model_cfg.TEACHER_HEAD)
        
        # Student branch (pre-pruning)
        self.student_backbone = build_backbone(model_cfg.STUDENT_BACKBONE)
        self.student_head = build_dense_head(model_cfg.STUDENT_HEAD)
        
        # Pruning module
        self.pruning_module = CategoryAwarePruning(
            num_categories=num_class,
            in_channels=model_cfg.STUDENT_BACKBONE.NUM_FILTERS[-1]
        )
        
        # Feature projection modules for distillation
        self.teacher_projection = FeatureProjection(
            model_cfg.TEACHER_BACKBONE.NUM_FILTERS[-1],
            model_cfg.FEATURE_DISTILLATION.PROJECTION_CHANNELS
        )
        self.student_projection = FeatureProjection(
            model_cfg.STUDENT_BACKBONE.NUM_FILTERS[-1],
            model_cfg.FEATURE_DISTILLATION.PROJECTION_CHANNELS
        )

        self.loss_weights = model_cfg.LOSS_WEIGHTS
 
        self.label_assigner = LabelAssignment(
            iou_threshold=model_cfg.LABEL_ASSIGNMENT.IOU_THRESHOLD
        )
        self.soft_target_generator = SoftTargetGenerator(
            temperature=model_cfg.KD.TEMPERATURE
        )
        
    def forward(self, batch_dict):
        # Teacher forward pass
        teacher_features = self.teacher_backbone(batch_dict)
        teacher_dict = self.teacher_head(teacher_features)
        
        # Student forward pass (pre-pruning)
        student_features = self.student_backbone(batch_dict)
        student_dict = self.student_head(student_features)

        pruned_features, channel_importance = self.pruning_module(
            student_features['spatial_features'],
            student_dict['cls_preds']
        )
        
        # Update student features with pruned features
        student_features['spatial_features'] = pruned_features
        
        # Pruned student forward pass
        pruned_student_dict = self.student_head(student_features)
        
        # Feature distillation
        teacher_proj = self.teacher_projection(teacher_features['spatial_features'])
        student_proj = self.student_projection(pruned_features)
        
        # Add label assignment and soft target generation
        teacher_matched_idx, student_matched_idx = self.label_assigner(
            teacher_dict['box_preds'],
            pruned_student_dict['box_preds'],
            batch_dict['gt_boxes']
        )
        
        soft_cls_targets, soft_reg_targets = self.soft_target_generator(
            teacher_dict['cls_preds'],
            teacher_dict['box_preds'],
            teacher_matched_idx
        )
        
        # Prepare output dictionary
        ret_dict = {
            'teacher_cls_preds': teacher_dict['cls_preds'],
            'teacher_box_preds': teacher_dict['box_preds'],
            'student_cls_preds': student_dict['cls_preds'],
            'student_box_preds': student_dict['box_preds'],
            'pruned_student_cls_preds': pruned_student_dict['cls_preds'],
            'pruned_student_box_preds': pruned_student_dict['box_preds'],
            'teacher_features': teacher_proj,
            'student_features': student_proj,
            'channel_importance': channel_importance,
            'teacher_matched_idx': teacher_matched_idx,
            'student_matched_idx': student_matched_idx,
            'soft_cls_targets': soft_cls_targets,
            'soft_reg_targets': soft_reg_targets
        }
        
        return ret_dict
        
    def get_training_loss(self):
        # Get supervised losses
        cls_loss, reg_loss = self.student_head.get_loss()
        
        # Feature distillation loss
        feat_kd_loss = torch.nn.functional.mse_loss(
            self.forward_ret_dict['student_features'],
            self.forward_ret_dict['teacher_features']
        )
        
        # Label distillation losses with soft targets
        cls_kd_loss = 0
        reg_kd_loss = 0
        
        for b in range(len(self.forward_ret_dict['soft_cls_targets'])):
            # Classification KD loss
            student_cls = self.forward_ret_dict['pruned_student_cls_preds'][b]
            soft_cls = self.forward_ret_dict['soft_cls_targets'][b]
            cls_kd_loss += torch.nn.functional.kl_div(
                torch.log_softmax(student_cls, dim=1),
                soft_cls,
                reduction='batchmean'
            )
            
            # Regression KD loss
            student_reg = self.forward_ret_dict['pruned_student_box_preds'][b]
            soft_reg = self.forward_ret_dict['soft_reg_targets'][b]
            reg_kd_loss += torch.nn.functional.mse_loss(student_reg, soft_reg)
        
        # Average over batch
        cls_kd_loss = cls_kd_loss / len(self.forward_ret_dict['soft_cls_targets'])
        reg_kd_loss = reg_kd_loss / len(self.forward_ret_dict['soft_reg_targets'])
        
        # Combine losses
        loss = (
            self.loss_weights.SUPERVISED_CLS * cls_loss +
            self.loss_weights.SUPERVISED_REG * reg_loss +
            self.loss_weights.FEAT_KD * feat_kd_loss +
            self.loss_weights.CLS_KD * cls_kd_loss +
            self.loss_weights.REG_KD * reg_kd_loss
        )
        
        tb_dict = {
            'loss': loss.item(),
            'cls_loss': cls_loss.item(),
            'reg_loss': reg_loss.item(),
            'feat_kd_loss': feat_kd_loss.item(),
            'cls_kd_loss': cls_kd_loss.item(),
            'reg_kd_loss': reg_kd_loss.item()
        }
        
        return loss, tb_dict 