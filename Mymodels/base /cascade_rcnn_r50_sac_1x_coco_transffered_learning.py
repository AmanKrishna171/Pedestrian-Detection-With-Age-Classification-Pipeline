_base_ = [
    '/home/a/t/test/PedesFormer-Transformer-Networks-For-Pedestrian-Detection/configs/_base_/models/detect_mine.py',
   '/home/a/Pedestrian-Detetction-with-Age-Classification/Project_Code/base_models/cityscapes_detection.py',
     '/home/a/Pedestrian-Detetction-with-Age-Classification/Project_Code/base_models/schedule_1x.py', '/home/a/Pedestrian-Detetction-with-Age-Classification/Project_Code/base_models/default_runtime.py'


model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True)))
