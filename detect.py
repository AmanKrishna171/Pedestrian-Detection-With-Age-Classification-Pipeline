import copy
import re
from utils.datasets import Get_Dataset
import model as models
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch
import sys
import time
import shutil
import argparse
import glob
import os
from mmcv.utils import is_str
from enum import Enum, Flag
from mmcv.image import imread, imwrite
from argparse import ArgumentParser
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv
import cv2
import argparse

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Detect Pedestrian')

parser.add_argument("-i", "--image", help="Prints the supplied argument.")
parser.add_argument("-c", "--config", help="Prints the supplied argument.")
parser.add_argument("-m", "--model", help="Prints the supplied argument.")

args = parser.parse_args()


config_file = '/home/a/Pedestrian-Detetction-with-Age-Classification/Project_Code/testmodel/2headSAC/detectors_cascade_rcnn_r50_1x_coco_2_heads.py'
checkpoint_file = '/home/a/Pedestrian-Detetction-with-Age-Classification/Project_Code/testmodel/2headSAC/epoch_20_2head.pth'

device = 'cuda:0'



if(args.image is not None):  # if image is supplied
    img = args.image
else:                       # if image is not supplied
    img = './1.png'
image = cv2.imread(img)

if(args.config is not None):  # if config is supplied
    config_file = args.config
else:                       # if config is not supplied
    config_file = './1.png'
image = cv2.imread(img)

if(args.model is not None):  # if model is supplied
    checkpoint_file = args.model
else:                       # if model is not supplied
    checkpoint_file = './1.png'
image = cv2.imread(img)


GLOBAL_ADULT_INDEX = []
GLOBAL_NON_ADULT_INDEX = []

model = init_detector(config_file, checkpoint_file,
                      device=device)  # init a detector

result = inference_detector(model, "test1.png")  # test the detector

print("Model succesfully loaded\n\n\n")


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241), out_file='result.jpg')


class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='red',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None,
                      show_box=True,
                      show_type='person'):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)
    img = np.ascontiguousarray(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    if (show_type == "person"):  # deafult
        bbox_color = color_val(bbox_color)
        text_color = color_val(text_color)

    elif(show_type == 'above_16'):  # if above_16 show green boxes with green text
        bbox_color = color_val("green")
        text_color = color_val("green")
    else:
        # if below_16 show red thick boxes with red text
        bbox_color = color_val("red")
        text_color = color_val("red")
        thickness = 4

    for bbox, label in zip(bboxes, labels):

        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])

        if(show_box and bbox[4] > 0.50):

            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            label_text = class_names[
                label] if class_names is not None else f'cls {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img


def show_mask_result(img, result, save_img, dataset='coco', score_thr=0.7, with_mask=False, showbox=False,
                     show_type='person'):
    segm_result = None
    if with_mask:
        bbox_result, segm_result = result
    else:
        bbox_result = result
    if isinstance(dataset, str):  # add own data label to mmdet.core.class_name.py
        class_names = [show_type]
        # print(class_names)
    elif isinstance(dataset, list):
        class_names = dataset
    else:
        raise TypeError('dataset must be a valid dataset name or a list'
                        ' of class names, not {}'.format(type(dataset)))
    h, w, _ = img.shape
    img_show = img[:h, :w, :]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]

    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)

    formated_result = (bboxes, labels, img_show)

    if(showbox):
        imshow_det_bboxes(img_show, bboxes, labels, class_names=class_names,
                          score_thr=score_thr, show=False, out_file=save_img, show_box=showbox, show_type=show_type)
    print("Output image saved to {}".format(save_img),"\n")                      
    return formated_result


t = 0
print("Detecting Pedestrains....\n")

import time
 

start = time.time()

result = inference_detector(model, img)

end = time.time()
print("The Detector Took :", end-start, "seconds\n")

# if len(result) > 1:  # uncoment for models with masks
#     result = result[0]


image = cv2.imread(img)

# the code below is for only showing the pedestrians
counter = 0
for i in result:
    if(counter > 0):
        result[counter] = np.zeros((0, 5))

    counter += 1


#  the code belwo is tp filter predictions with low confidence score

temp_result_1 = copy.copy(result)  # make a shallow copy of the result

temp_result_1 = temp_result_1[0].tolist()  # convert the numpy array to a list

output_result_3 = []  # output list

for i in range(len(temp_result_1)):
    j = temp_result_1[i]
    if (j[4] > 0.68):
        output_result_3.append(temp_result_1[i])

result[0] = np.array(output_result_3)

# print (result)
# check here for errror


formated_result = show_mask_result(image, result, 'result.png', showbox=True)
files = glob.glob('data_path/detected_image/*.png')

# clear all images froom folder
for i in files:
    os.remove(i)
f = open('data_path/detected_image/image.txt', 'w')

image=cv2.imread(img)

for bbox, label in zip(formated_result[0], formated_result[1]):
    bbox_int = bbox.astype(np.int32)

    if (bbox[4] > 0.68):
        # get the bounding box values
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])

        t1 = left_top[0] - 0
        t2 = left_top[1] - 0
        t3 = right_bottom[0] +0
        t4 = right_bottom[1] +0

        # crop the image using the bounding box values
        roi_cropped = image[int(t2):int(t4), int(t1):int(t3)]

        # save the cropped image and add annotation to the file
        filename = "detected_image/" + str(t)+"_crop.png"
        cv2.imwrite('data_path/'+filename, roi_cropped)
        f.write(filename+'\n')

        t = t+1

f.close()

print("Pedestrain Detection Model has detected ", t, " pedestrains in ", img)

print("Detected Files stored in ./detected_image/\n\n")

total_confiednce = 0

for bbox, label in zip(formated_result[0], formated_result[1]):
    bbox_int = bbox.astype(np.int32)

    total_confiednce= bbox[4] + total_confiednce

print("Average Confidence is : ", total_confiednce/len(result[0]),'\n\n')


def classify(data_model):
    detected = {'above_16': 0, "less_than_16": 0}

    input_dataset, attr_num, description = Get_Dataset(
        data_model, 'inception_iccv')
    # get input images and labels

    val_loader = torch.utils.data.DataLoader(
        input_dataset,
        batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    # convert dataset to pytorch dataset type

    model = models.__dict__['inception_iccv'](
        pretrained=True, num_classes=attr_num)
    # load pretrained model

    model = torch.nn.DataParallel(model).cuda()
    # parallelize model

    if os.path.isfile("rap_epoch_9.pth.tar"):
        # print("=> loading checkpoint '{}'".format("./rap_epoch_9.pth.tar"))
        checkpoint = torch.load("rap_epoch_9.pth.tar")

        best_accu = checkpoint['best_accu']
        model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {})"
        #   .format("./rap_epoch_9.pth.tar", checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(
            "./rap_epoch_9.pth.pth.tar"))

    cudnn.benchmark = False
    cudnn.deterministic = True

    # define loss function
    criterion = Weighted_BCELoss('rap')

    if True:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                     betas=(0.9, 0.999),
                                     weight_decay=0.0005)

    print("Classifying detected pedestrains....\n")

    result = test(val_loader, model, attr_num, description)

    # get the result
    counter = 0
    for i in result:
        if (i[1] == 1):
            detected['less_than_16'] = detected['less_than_16'] + 1
            GLOBAL_NON_ADULT_INDEX.append(counter)
        else:
            detected['above_16'] = detected['above_16'] + 1
            GLOBAL_ADULT_INDEX.append(counter)
        counter += 1
    return detected


def test(val_loader, model, attr_num, description):
    model.eval()

    pos_cnt = []
    pos_tol = []
    neg_cnt = []
    neg_tol = []

    accu = 0.0
    prec = 0.0
    recall = 0.0
    tol = 0

    for it in range(attr_num):
        pos_cnt.append(0)
        pos_tol.append(0)
        neg_cnt.append(0)
        neg_tol.append(0)

    for i, _ in enumerate(val_loader):
        input, target = _
        target = target.cuda(non_blocking=True)

        input = input.cuda(non_blocking=True)
        output = model(input)

        bs = target.size(0)

        # maximum voting
        if type(output) == type(()) or type(output) == type([]):
            output = torch.max(
                torch.max(torch.max(output[0], output[1]), output[2]), output[3])

        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)

        # for it in range(attr_num):
        #     for jt in range(batch_size):
        #         if target[jt][it] == 1:
        #             pos_tol[it] = pos_tol[it] + 1
        #             if output[jt][it] == 1:
        #                 pos_cnt[it] = pos_cnt[it] + 1
        #         if target[jt][it] == 0:
        #             neg_tol[it] = neg_tol[it] + 1
        #             if output[jt][it] == 0:
        #                 neg_cnt[it] = neg_cnt[it] + 1

        # if attr_num == 1:
        #     continue
        # for jt in range(batch_size):
        #     tp = 0
        #     fn = 0
        #     fp = 0
        #     for it in range(attr_num):
        #         if output[jt][it] == 1 and target[jt][it] == 1:
        #             tp = tp + 1
        #         elif output[jt][it] == 0 and target[jt][it] == 1:
        #             fn = fn + 1
        #         elif output[jt][it] == 1 and target[jt][it] == 0:
        #             fp = fp + 1
        #     if tp + fn + fp != 0:
        #         accu = accu +  1.0 * tp / (tp + fn + fp)
        #     if tp + fp != 0:
        #         prec = prec + 1.0 * tp / (tp + fp)
        #     if tp + fn != 0:
        #         recall = recall + 1.0 * tp / (tp + fn)

    # print('=' * 100)
    # print('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
    # mA = 0.0
    # for it in range(attr_num):
    #     cur_mA = ((1.0*pos_cnt[it]/pos_tol[it]) + (1.0*neg_cnt[it]/neg_tol[it])) / 2.0
    #     mA = mA + cur_mA
    #     print('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,description[it],pos_cnt[it],neg_cnt[it],pos_tol[it],neg_tol[it],(pos_cnt[it]+neg_tol[it]-neg_cnt[it]),(neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA))
    # mA = mA / attr_num
    # print('\t' + 'mA:        '+str(mA))
    
    return output


class Weighted_BCELoss(object):
    """
        Weighted_BCELoss was proposed in "Multi-attribute learning for pedestrian attribute recognition in surveillance scenarios"[13].
    """

    def __init__(self, experiment):
        super(Weighted_BCELoss, self).__init__()
        self.weights = None
        if experiment == 'pa100k':
            self.weights = torch.Tensor([0.460444444444,
                                        0.0134555555556,
                                        0.924377777778,
                                        0.0621666666667,
                                        0.352666666667,
                                        0.294622222222,
                                        0.352711111111,
                                        0.0435444444444,
                                        0.179977777778,
                                        0.185,
                                        0.192733333333,
                                        0.1601,
                                        0.00952222222222,
                                        0.5834,
                                        0.4166,
                                        0.0494777777778,
                                        0.151044444444,
                                        0.107755555556,
                                        0.0419111111111,
                                        0.00472222222222,
                                        0.0168888888889,
                                        0.0324111111111,
                                        0.711711111111,
                                        0.173444444444,
                                        0.114844444444,
                                        0.006]).cuda()
        elif experiment == 'rap':
            self.weights = torch.Tensor([0.311434,
                                        0.009980,
                                        0.430011,
                                        0.560010,
                                        0.144932,
                                        0.742479,
                                        0.097728,
                                        0.946303,
                                        0.048287,
                                        0.004328,
                                        0.189323,
                                        0.944764,
                                        0.016713,
                                        0.072959,
                                        0.010461,
                                        0.221186,
                                        0.123434,
                                        0.057785,
                                        0.228857,
                                        0.172779,
                                        0.315186,
                                        0.022147,
                                        0.030299,
                                        0.017843,
                                        0.560346,
                                        0.000553,
                                        0.027991,
                                        0.036624,
                                        0.268342,
                                        0.133317,
                                        0.302465,
                                        0.270891,
                                        0.124059,
                                        0.012432,
                                        0.157340,
                                        0.018132,
                                        0.064182,
                                        0.028111,
                                        0.042155,
                                        0.027558,
                                        0.012649,
                                        0.024504,
                                        0.294601,
                                        0.034099,
                                        0.032800,
                                        0.091812,
                                        0.024552,
                                        0.010388,
                                        0.017603,
                                        0.023446,
                                        0.128917]).cuda()
        elif experiment == 'peta':
            self.weights = torch.Tensor([0.5016,
                                        0.3275,
                                        0.1023,
                                        0.0597,
                                        0.1986,
                                        0.2011,
                                        0.8643,
                                        0.8559,
                                        0.1342,
                                        0.1297,
                                        0.1014,
                                        0.0685,
                                        0.314,
                                        0.2932,
                                        0.04,
                                        0.2346,
                                        0.5473,
                                        0.2974,
                                        0.0849,
                                        0.7523,
                                        0.2717,
                                        0.0282,
                                        0.0749,
                                        0.0191,
                                        0.3633,
                                        0.0359,
                                        0.1425,
                                        0.0454,
                                        0.2201,
                                        0.0178,
                                        0.0285,
                                        0.5125,
                                        0.0838,
                                        0.4605,
                                        0.0124]).cuda()
        #self.weights = None


final_result = classify('rap')
print('\n')
print("Pedestrain Classification Model has detected ",
      final_result['above_16'], " above_16 and ", final_result['less_than_16'], " person below 16.")
print('\n')


image = cv2.imread(img)

if(len(GLOBAL_ADULT_INDEX) > 0):
    adult_result = copy.copy(result)

    temp_result_2 = adult_result[0].tolist()
    temp_result_3 = []

    for i in range(len(temp_result_2)):
        j = temp_result_2[i]
        if (i in GLOBAL_ADULT_INDEX):
            temp_result_3.append(temp_result_2[i])

    adult_result[0] = np.array(temp_result_3)
    show_mask_result(image, adult_result, 'result1.png',
                     showbox=True, show_type="above_16")


#  now red bounding box for the pedestrians belwo 16 are drawn in output image
if(len(GLOBAL_NON_ADULT_INDEX) > 0):
    non_adult_result = copy.copy(result)

    temp_result_2 = non_adult_result[0].tolist()
    temp_result_3 = []

    for i in range(len(temp_result_2)):  # adult pedestrians are filteredp out
        j = temp_result_2[i]
        if (i in GLOBAL_NON_ADULT_INDEX):
            temp_result_3.append(temp_result_2[i])

    # new list without adult pedestrians is created
    non_adult_result[0] = np.array(temp_result_3)

    show_mask_result(image, non_adult_result, 'result1.png',
                     showbox=True, show_type='below_16')  # the bounding box is drawn
