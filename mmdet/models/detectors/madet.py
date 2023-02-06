from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class MADet(SingleStageDetector):
    """
        Detector for MADet
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(MADet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
