import torch
import torch.nn as nn
import ipdb
import numpy as np
from numpy import *
import torch.nn.functional as F
from torchvision.ops import roi_align
import random


from ...utils import common_utils, box_utils, loss_utils, box_coder_utils
from .roi_head_template import RoIHeadTemplate
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ..model_utils.helper import  _topk, _nms, extract_input_from_tensor
from ..model_utils.trans_mm import build_transformer
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from ...ops.pointnet2.pointnet2_stack.pointnet2_utils import CylinderQuery, GroupingOperation

from ..model_utils.trans_channel_wise import build_transformer_cw
# from ..model_utils.resnet import build_resnet
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch

# from pcdet.datasets.augmentor.data_augmentor import random_world_scaling_rcnn

cylinder_query = CylinderQuery.apply
grouping_operation = GroupingOperation.apply

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CT3DPlusPlusHead(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, voxel_size, point_cloud_range, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        # LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = model_cfg.ROI_GRID_POOL.Semantic_dim
        self.semantic_embed = MLP(256, 64, c_out, 3)
        self.fuse_embed = MLP(c_out+1+27, 256, 256, 3)
        # self.fuse_embed = nn.Linear(32+256, 256)

        # ipdb.set_trace()
        self.c_out = c_out
        num_queries = model_cfg.Transformer.num_queries
        hidden_dim = model_cfg.Transformer.hidden_dim
        self.num_points = model_cfg.Transformer.num_points

        self.class_embed = nn.Linear(hidden_dim, 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, self.box_coder.code_size * self.num_class, 4)
        self.jud_embed = MLP(64, 256, 1, 2)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = build_transformer(model_cfg.Transformer)
        self.trans_cw = build_transformer_cw()
        self.aux_loss = model_cfg.Transformer.aux_loss

        self.car_radii = 2.6

        self.init_weights(weight_init='xavier')


    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.bbox_embed.layers[-1].weight, mean=0, std=0.001)

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'jud_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )

    def get_global_grid_points_of_roi(self, rois):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_corner_points(rois, batch_size_rcnn)  # (BxN, 2x2x2, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        # ipdb.set_trace()

        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_corner_points(rois, batch_size_rcnn):
        faked_features = rois.new_ones((2, 2, 2))

        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 2x2x2, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 2x2x2, 3)
        return roi_grid_points
    
    def spherical_coordinate(self, src, diag_dist):
        assert (src.shape[-1] == 27)
        device = src.device
        indices_x = torch.LongTensor([0,3,6,9,12,15,18,21,24]).to(device)  #
        indices_y = torch.LongTensor([1,4,7,10,13,16,19,22,25]).to(device) # 
        indices_z = torch.LongTensor([2,5,8,11,14,17,20,23,26]).to(device) 
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / diag_dist
        src = torch.cat([dis, phi, the], dim = -1)
        return src

    def interpolate_from_bev_features(self, points, bev_features, batch_size, bev_stride):
        x_idxs = (points[:,:,1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (points[:,:,2] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features

    def get_clsreg_targets(self, forward_ret_dict):
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0
        # ipdb.set_trace()
        reg_targets = self.box_coder.encode_torch(
            gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
        )
        bbox1 = rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0)
        bbox2 = reg_targets.unsqueeze(dim=0)


        _, rdiou = self.get_rdiou(bbox1, bbox2)

        with torch.no_grad():
            reg_rdiou_for_conf =  rdiou.detach()
        return reg_rdiou_for_conf

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels_s1 = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_cls_labels_s2 = self.reg_rdiou_for_conf.view(-1)
        # rcnn_cls_labels = rcnn_cls_labels_s1*0.7 + rcnn_cls_labels_s2*0.3
        rcnn_cls_labels = rcnn_cls_labels_s1
        fg_mask = (rcnn_cls_labels_s1 >= 0).float()
        # fg_mask = ((rcnn_cls_labels > 0.1) & (rcnn_cls_labels_s1 >= 0)).float()

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            # cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * fg_mask).sum() / torch.clamp(fg_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            # cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * fg_mask).sum() / torch.clamp(fg_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            # ipdb.set_trace()
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()

        elif loss_cfgs.REG_LOSS == 'rdiou':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            # ipdb.set_trace()
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )
            bbox1 = rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0)
            bbox2 = reg_targets.unsqueeze(dim=0)

            u, rdiou = self.get_rdiou(bbox1, bbox2)
            rdiou_loss_n = rdiou - u
            rdiou_loss_n = torch.clamp(rdiou_loss_n,min=-1.0,max = 1.0)
            rdiou_loss_m = 1 - rdiou_loss_n
            rcnn_loss_reg = (rdiou_loss_m.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
            
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict


    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
        # if True:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        
        # ipdb.set_trace()
        
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        num_rois = batch_dict['rois'].shape[-2]

        # corner
        corner_points, _ = self.get_global_grid_points_of_roi(rois)  # (BxN, 2x2x2, 3)
        corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])  # (B, N, 2x2x2, 3)
        corner_points = corner_points.view(batch_size * num_rois, -1)
        corner_add_center_points = torch.cat([corner_points, rois.view(-1, rois.shape[-1])[:,:3]], dim = -1)

        num_sample = self.num_points
        # get raw points and features
        features = batch_dict['points'][:, 4:5].contiguous()
        xyz = batch_dict['points'][:, 1:4].contiguous()
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = batch_dict['points'][:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()
        new_xyz = rois.view(-1, rois.shape[-1])[:,:3].contiguous()
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(rois.shape[1])
        idx, empty_cylinder_mask = cylinder_query(self.car_radii, num_sample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz[empty_cylinder_mask] = 0
        grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
        grouped_features[empty_cylinder_mask] = 0
        src = torch.cat([grouped_xyz, grouped_features], dim=1).permute(0,2,1)   # (b*128, nsample, 4)

        # point aware extraction
        point = torch.cat([src[:,:,:3], corner_add_center_points.view(batch_size*num_rois, 9, 3)], dim=-2).view(batch_size, -1, 3)
        if 'bev' in self.pool_cfg.FEATURES_SOURCE:
            extracted_multi_scale_features = self.interpolate_from_bev_features(
                point, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
        extracted_multi_scale_features = extracted_multi_scale_features.view(batch_size * num_rois, -1, batch_dict['spatial_features'].shape[1])
        extracted_multi_scale_features = self.semantic_embed(extracted_multi_scale_features)
        # ipdb.set_trace()

        # proposal-to-point geometric
        pos_fea = torch.cat([src[:,:,:3].repeat(1,1,9), corner_add_center_points.unsqueeze(1).repeat(1,9,1)], dim = 1)
        pos_fea = pos_fea - corner_add_center_points.unsqueeze(1).repeat(1,num_sample+9,1)
        lwh = rois.view(-1, rois.shape[-1])[:,3:6].unsqueeze(1).repeat(1,num_sample+9,1)
        diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
        pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))
        src_lidar = pos_fea

        # proposal-to-point semantic
        key_zero_reflect = src.new_zeros(src.shape[0], 9, 1)
        reflect = torch.cat([src[:,:,2:3], key_zero_reflect], dim = 1)
        src_mc = torch.cat([extracted_multi_scale_features.view(batch_size*num_rois, num_sample+9, -1),reflect], dim =-1)

        # ipdb.set_trace()

        # up dimension
        # src_lidar_ = self.lidar_embed(src_lidar)
        src = self.fuse_embed(torch.cat([src_lidar, src_mc], dim=-1))
        # src = self.trans_cw(src_lidar, src_mc)
        hs = self.transformer(src[:,:num_sample,:], src[:,num_sample:,:], self.query_embed.weight)

        # output
        rcnn_cls = self.class_embed(hs[0])[-1].squeeze(0)
        rcnn_reg = self.bbox_embed(hs[0])[-1].squeeze(0)

        # ipdb.set_trace()
    
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_size, rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_cls_preds = (batch_cls_preds+targets_dict['roi_scores'].view(batch_cls_preds.shape))/2
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.reg_rdiou_for_conf = self.get_clsreg_targets(targets_dict)
            self.forward_ret_dict = targets_dict

        return batch_dict