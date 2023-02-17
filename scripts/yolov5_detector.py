# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt # For WARNING: QApplication was not created in the main() thread.

import os
import sys
cwd = os.getcwd().rstrip('scripts')
sys.path.append(os.path.join(cwd, 'modules/yolov5-test'))

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import argparse
import time
from pathlib import Path
import numpy as np
from numpy import random

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

def draw_predictions(img, label, score, box, color=(156, 39, 176), location=None):
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.5
    f_thickness, l_thickness = 1, 2
    
    h, w, _ = img.shape
    u1, v1, u2, v2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (u1, v1), (u2, v2), color, l_thickness)
    
    text = '%s: %.2f' % (label, score)
    text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
    text_h += 6
    if v1 - text_h < 0:
        cv2.rectangle(img, (u1, text_h), (u1 + text_w, 0), color, -1)
        cv2.putText(img, text, (u1, text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (u1, v1), (u1 + text_w, v1 - text_h), color, -1)
        cv2.putText(img, text, (u1, v1 - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    
    if location is not None:
        text = '(%.1fm, %.1fm)' % (location[0], location[1])
        text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
        text_h += 6
        if v2 + text_h > h:
            cv2.rectangle(img, (u1, h - text_h), (u1 + text_w, h), color, -1)
            cv2.putText(img, text, (u1, h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (u1, v2), (u1 + text_w, v2 + text_h), color, -1)
            cv2.putText(img, text, (u1, v2 + text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    
    return img

class Yolov5Detector():
    def __init__(self, weights=''):
        imgsz = 640
        self.device = device = select_device('')
        self.half = half = device.type != 'cpu' # half precision only supported on CUDA
        
        # Load model
        weights = os.path.join(cwd, 'modules/yolov5-test', weights)
        self.model = model = attempt_load(weights, map_location=device) # load FP32 model
        self.stride = stride = int(model.stride.max()) # model stride
        self.imgsz = imgsz = check_img_size(imgsz, s=stride) # check img_size
        if half:
            model.half() # to FP16
        
        # Get names
        self.names = model.module.names if hasattr(model, 'module') else model.names
        
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once
    
    def run(self, img0, conf_thres=0.25, iou_thres=0.45, classes=None):
        """
        Args:
            img0: (h, w, 3), BGR format
            conf_thres: float, object confidence threshold
            iou_thres: float, IOU threshold for NMS
            classes: list(int), filter by class, for instance [0, 2, 3]
        Returns:
            labels: list(str)
            scores: list(float)
            boxes: (n, 4), xyxy format
        """
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float() # uint8 to fp16/32
        img /= 255.0 # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        pred = self.model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)
        
        # Process detections
        det = pred[0]
        if len(det):
            # Rescale boxes from imgsz to img0 size
            boxes = scale_coords(img.shape[2:], det[:, :4], img0.shape).round().cpu().numpy() # xyxy
            labels = [self.names[int(cls)] for cls in det[:, -1]]
            scores = [float('%.2f' % conf) for conf in det[:, -2]]
            return labels, scores, boxes
        else:
            return [], [], np.array([])

if __name__ == '__main__':
    file_name = '/home/lishangjie/data/SEUMM/seumm_visible/images/001000.jpg'
    #~ file_name = '/home/lishangjie/data/SEUMM/seumm_lwir/images/001000.jpg'
    assert os.path.exists(file_name), '%s Not Found' % file_name
    img = cv2.imread(file_name)
    
    detector = Yolov5Detector(weights='weights/seumm_visible/yolov5s_100ep_pretrained.pt')
    #~ detector = Yolov5Detector(weights='weights/seumm_lwir/yolov5s_100ep_pretrained.pt')
    
    t1 = time.time()
    labels, scores, boxes = detector.run(
        img, classes=[0, 1, 2, 3, 4]
    ) # pedestrian, cyclist, car, bus, truck
    t2 = time.time()
    print('time cost:', t2 - t1, '\n')
    
    print('labels', type(labels), len(labels))
    print('scores', type(scores), len(scores))
    print('boxes', type(boxes), boxes.shape)
    
    for i in range(len(labels)):
        img = draw_predictions(img, str(labels[i]), float(scores[i]), boxes[i])
    
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    
    if cv2.waitKey(0) == 27:
        cv2.destroyWindow("img")
