from mmdet.apis import init_detector, inference_detector

config_file = 'configs/madet/madet_r50_fpn_1x_coco.py'
# put the trained detection model in `checkpoints/`
checkpoint_file = 'checkpoints/model.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
inference_detector(model, 'demo/demo.jpg')