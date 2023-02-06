import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap, build_bbox_coder, build_anchor_generator,
                        distance2bbox)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from mmdet.core.bbox.iou_calculators import bbox_overlaps

EPS = 1e-12


def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@HEADS.register_module()
class MAHead(BaseDenseHead, BBoxTestMixin):
    """
        Detection head for MADet
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=3,
                 feat_channels=256,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 anchor_generator=dict(
                    type='AnchorGenerator',
                    ratios=[1.0],
                    octave_base_scale=8,
                    scales_per_octave=1,
                    strides=[8, 16, 32, 64, 128]),
                 bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                 loss_cls=dict(
                    type='WeightedFocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                 loss_bbox=dict(
                    type='GIoULoss', 
                    loss_weight=0.5),
                 loss_iou=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.5),
                 train_cfg=None,
                 test_cfg=None):
        super(MAHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss']
        
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_denoms = [64, 128, 256, 512, 1024]

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self._init_layers()
        
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.class_conv = nn.Conv2d(
            self.feat_channels,
            2*self.cls_out_channels,
            3,
            padding=1)
        self.regress_conv_abased = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)
        self.regress_conv_afree = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)
        self.iou_conv = nn.Conv2d(
            self.feat_channels, 2, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])
        self.dcn = ConvModule(
            2*self.feat_channels,
            2*self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='DCNv2'),
            norm_cfg=self.norm_cfg)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.class_conv, std=0.01, bias=bias_cls)
        normal_init(self.regress_conv_abased, std=0.01)
        normal_init(self.regress_conv_afree, std=0.01)
        normal_init(self.iou_conv, std=0.01)
        normal_init(self.dcn, std=0.01)

    def fian(self, input_feat):
        """
            Feature Interactive Alignment Network
        """
        cls_feat = input_feat
        reg_feat = input_feat
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        feat = torch.cat((cls_feat, reg_feat), dim=1)
        if feat.shape[2] < 3:
            feat = F.pad(feat, (0, 0, 0, 3-feat.shape[2]), mode='constant', value=0)
        if feat.shape[3] < 3:
            feat = F.pad(feat, (0, 3-feat.shape[3]), mode='constant', value=0)
        feat = self.dcn(feat)
        cls_feat = feat[:, :self.feat_channels, ...]
        reg_feat = feat[:, self.feat_channels:, ...]
        return cls_feat, reg_feat

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales, self.reg_denoms)

    def forward_single(self, x, scale, reg_denom):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is 2 * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is 2 * 4.
                iou_pred (Tensor): iou_pred for a single scale level, the
                    channel number is (N, 2, H, W).
        """
        cls_feat, reg_feat = self.fian(x)

        cls_score = self.class_conv(cls_feat)

        # we don't apply exp in bbox_pred_abased, but apply exp in bbox_pred_afree
        bbox_pred_abased = scale(self.regress_conv_abased(reg_feat)).float()
        bbox_pred_afree = scale(self.regress_conv_afree(reg_feat)).float().exp() * reg_denom
        bbox_pred = torch.cat((bbox_pred_abased, bbox_pred_afree), dim=1)
        
        iou_pred = self.iou_conv(reg_feat)

        return cls_score, bbox_pred, iou_pred

    def loss_single(self, anchors, cls_score, bbox_pred, iou_pred, labels,
                    label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, 2* num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, 8, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors, 2).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors, 2)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 8).
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert cls_score.shape[-2:] == bbox_pred.shape[-2:]

        anchors = anchors.reshape(-1, 4)

        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, 2 * self.cls_out_channels)
        cls_score_abased = cls_score[:, :self.cls_out_channels]
        cls_score_afree = cls_score[:, self.cls_out_channels:]
        cls_score = torch.cat((cls_score_abased, cls_score_afree), dim=0).contiguous()

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 8)

        bbox_targets = bbox_targets.reshape(-1, 8)

        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1, 2)
        iou_pred_abased = iou_pred[:, 0].unsqueeze(1)
        iou_pred_abased = iou_pred_abased.sigmoid()
        iou_pred_afree = iou_pred[:, 1].unsqueeze(1)
        iou_pred_afree = iou_pred_afree.sigmoid()

        labels = labels.reshape(-1, 2)

        rank_score_abased = cls_score_abased.sigmoid() * iou_pred_abased
        rank_score_afree = cls_score_afree.sigmoid() * iou_pred_afree

        beta = 2
        t = lambda x: 1 / (0.5 ** beta - 1) * x ** beta - 1 / (0.5 ** beta - 1)

        def normalize(x):
            x_ = t(x)
            t1 = x_.min()
            t2 = min(1., x_.max())
            y = (x_ - t1 + EPS) / (t2 - t1 + EPS)
            y[x < 0.5] = 1
            return y

        pos_weights_abased = torch.exp(rank_score_abased)
        pos_weights_afree = torch.exp(rank_score_afree)
        pos_weights = torch.cat((pos_weights_abased, pos_weights_afree))

        neg_weights_abased = torch.exp(normalize(iou_pred_abased) * rank_score_abased)
        neg_weights_afree = torch.exp(normalize(iou_pred_afree) * rank_score_afree)
        neg_weights = torch.cat((neg_weights_abased, neg_weights_afree))

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels[:, 0] >= 0)
                    & (labels[:, 0] < bg_class_ind)).nonzero().squeeze(1)

        # use weighted_focal_loss to calulate the classification loss
        labels_abased = labels[:, 0]
        cls_targets = torch.zeros_like(cls_score_abased)
        pos_labels = labels_abased[pos_inds]
        cls_targets[pos_inds, pos_labels] = torch.ones((pos_inds.size(0),),
                                                dtype=cls_targets.dtype, device=cls_targets.device)
        cls_targets = cls_targets.repeat(2, 1)
        loss_cls = self.loss_cls(cls_score,
                                 cls_targets,
                                 pos_weights,
                                 neg_weights,
                                 avg_factor=num_total_samples)

        if len(pos_inds) > 0:
            pos_bbox_targets_abased = bbox_targets[pos_inds, :4]
            pos_bbox_targets_afree = bbox_targets[pos_inds, 4:]

            pos_bbox_pred_abased = bbox_pred[pos_inds, :4]
            pos_bbox_pred_afree = bbox_pred[pos_inds, 4:]

            pos_anchors = anchors[pos_inds]
            pos_points = (pos_anchors[:, 0:2] + pos_anchors[:, 2:4]) / 2.0

            pos_decode_bbox_pred_abased = self.bbox_coder.decode(pos_anchors, pos_bbox_pred_abased)
            pos_decode_bbox_pred_afree = distance2bbox(pos_points, pos_bbox_pred_afree)

            pos_iou_pred = iou_pred[pos_inds].permute(1, 0).reshape(-1)

            iou_targets_abased = self.iou_target(pos_decode_bbox_pred_abased,
                                                 pos_bbox_targets_abased)
            iou_targets_afree = self.iou_target(pos_decode_bbox_pred_afree,
                                                pos_bbox_targets_afree)
            iou_targets = torch.cat((iou_targets_abased, iou_targets_afree))

            iou_weights_abased = iou_targets_abased / (iou_targets_abased + iou_targets_afree)
            iou_weights_afree = iou_targets_afree / (iou_targets_abased + iou_targets_afree)

            loss_weights_abased = torch.exp(iou_weights_abased) * iou_weights_abased
            loss_weights_afree = torch.exp(iou_weights_afree) * iou_weights_afree

            # regression loss
            loss_bbox_abased = self.loss_bbox(
                pos_decode_bbox_pred_abased,
                pos_bbox_targets_abased,
                weight=loss_weights_abased,
                avg_factor=1.0)
            loss_bbox_afree = self.loss_bbox(
                pos_decode_bbox_pred_afree,
                pos_bbox_targets_afree,
                weight=loss_weights_afree,
                avg_factor=1.0)
            loss_bbox = loss_bbox_abased + loss_bbox_afree

            # iou loss
            loss_iou = self.loss_iou(
                pos_iou_pred,
                iou_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_iou = iou_pred.sum() * 0
            iou_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_iou, iou_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, 2 * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 8, H, W)
            iou_preds (list[Tensor]): iou_pred for each scale
                level with shape (N, 2, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_scores_list = levels_to_images(cls_scores)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            cls_scores_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_iou, bbox_avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            iou_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
        if bbox_avg_factor < EPS:
            bbox_avg_factor = 1
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_iou=losses_iou)

    def iou_target(self, bbox_pred, bbox_targets):
        # only calculate pos iou targets
        return bbox_overlaps(bbox_pred, bbox_targets, mode='iou', is_aligned=True)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, 2 * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 2 * 4, H, W).
            iou_preds (list[Tensor]): iou_pred for each scale level with
                shape (N, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            iou_pred_list = [
                iou_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                iou_pred_list, mlvl_anchors, 
                                                img_shape, scale_factor, 
                                                cfg, rescale, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           iou_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (2 * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (2 * 4, H, W).
            iou_preds (list[Tensor]): IoU for a single scale level
                with shape (2, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_iou_preds = []
        for cls_score, bbox_pred, iou_pred, anchors in zip(
                cls_scores, bbox_preds, iou_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(-1, 2*self.cls_out_channels).sigmoid()
            scores_abased = scores[:, :self.cls_out_channels]
            scores_afree = scores[:, self.cls_out_channels:]
            scores = torch.cat((scores_abased, scores_afree), dim=0)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 8)
            bbox_pred_abased = bbox_pred[:, :4]
            bbox_pred_afree = bbox_pred[:, 4:]

            bboxes_abased = self.bbox_coder.decode(anchors, bbox_pred_abased, max_shape=img_shape)
            points = (anchors[:, 0:2] + anchors[:, 2:4]) / 2.0
            bboxes_afree = distance2bbox(points, bbox_pred_afree)
            bboxes = torch.cat((bboxes_abased, bboxes_afree), dim=0)

            iou_pred = iou_pred.permute(1, 2, 0).reshape(-1, 2).sigmoid()
            iou_pred = iou_pred.permute(1, 0).reshape(-1)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre*2:
                max_scores, _ = (scores * iou_pred[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre*2)
                bboxes = bboxes[topk_inds, :]
                scores = scores[topk_inds, :]
                iou_pred = iou_pred[topk_inds]

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_iou_preds.append(iou_pred)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_iou_preds = torch.cat(mlvl_iou_preds)
        mlvl_nms_scores = mlvl_scores * mlvl_iou_preds[:, None]

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_nms_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=None)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_iou_preds

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    cls_scores_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """
            Get targets for MADet head.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             cls_scores_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel()*2, 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel()*2, 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           cls_scores,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """
        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors (Tensor): Number of anchors of each scale level.
            cls_scores (Tensor): predicted classification scores of the image,
                shape (num_anchors, 2C).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N, 2).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N, 2).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 8).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 8)
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        valid_cls_scores = cls_scores[inside_flags, :self.cls_out_channels]

        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, valid_cls_scores,
                                             gt_bboxes, gt_bboxes_ignore, gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros((num_valid_anchors, 8),
                                    dtype=anchors.dtype,
                                    layout=anchors.layout,
                                    device=anchors.device)
        bbox_weights = torch.zeros((num_valid_anchors, 8),
                                    dtype=anchors.dtype,
                                    layout=anchors.layout,
                                    device=anchors.device)
        labels = anchors.new_full((num_valid_anchors, 2),
                                   self.num_classes,
                                   dtype=torch.long)
        label_weights = anchors.new_zeros((num_valid_anchors, 2), dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = torch.cat((sampling_result.pos_gt_bboxes,
                                                   sampling_result.pos_gt_bboxes), dim=1)
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds, :] = 0
            else:
                labels[pos_inds, 0] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
                labels[pos_inds, 1] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds, :] = 1.0
            else:
                label_weights[pos_inds, :] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds, :] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
