# -*- coding: UTF-8 -*-

import os
import sys
import rospy
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import threading
import argparse

from std_msgs.msg import Header
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from yolov5_dualdetector import Yolov5Detector, draw_predictions
from mono_estimator import MonoEstimator
from functions import get_stamp, publish_image
from functions import display, print_info
from functions import simplified_nms

parser = argparse.ArgumentParser(
    description='Demo script for dual modal peception')
parser.add_argument('--print', action='store_true',
    help='Whether to print and record infos.')
parser.add_argument('--sub_image1', default='/pub_rgb', type=str,
    help='The image topic to subscribe.')
parser.add_argument('--sub_image2', default='/pub_t', type=str,
    help='The image topic to subscribe.')
parser.add_argument('--pub_image', default='/result', type=str,
    help='The image topic to publish.')
parser.add_argument('--calib_file', default='../conf/calibration_image.yaml', type=str,
    help='The calibration file of the camera.')
parser.add_argument('--modality', default='dual', type=str,
    help='The modality to use. This should be `RGB`, `T` or `RGBT`.')
parser.add_argument('--indoor', action='store_true',
    help='Whether to use INDOOR detection mode.')
parser.add_argument('--frame_rate', default=10, type=int,
    help='Working frequency.')
parser.add_argument('--display', action='store_true',
    help='Whether to display and save all videos.')
args = parser.parse_args()
# 在线程函数执行前，“抢占”该锁，执行完成后，“释放”该锁，则我们确保了每次只有一个线程占有该锁。这也是为什么能让RGB和T匹配的原因
image1_lock = threading.Lock() #创建锁
image2_lock = threading.Lock()

#3.1 获取RGB图像的时间戳和格式转化
#×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
def image1_callback(image): 
    global image1_stamp, image1_frame #多线程是共享资源的，使用全局变量 
    image1_lock.acquire() #锁定锁
    image1_stamp = get_stamp(image.header) #获得时间戳
    image1_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)#图片格式转化
    image1_lock.release() #释放
#3.2 获取红外图像的时间戳和格式转化
#×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
def image2_callback(image):
    global image2_stamp, image2_frame
    image2_lock.acquire()
    image2_stamp = get_stamp(image.header)
    image2_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image2_lock.release()

#5.1 获取红外图像的时间戳和格式转化
#×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
def timer_callback(event):
    #获得RGB的时间戳和图片
    global image1_stamp, image1_frame
    image1_lock.acquire()
    cur_stamp1 = image1_stamp
    cur_frame1 = image1_frame.copy()
    image1_lock.release()
    #获得T的时间戳和图片
    global image2_stamp, image2_frame
    image2_lock.acquire()
    cur_stamp2 = image2_stamp
    cur_frame2 = image2_frame.copy()
    image2_lock.release()
    
    global frame
    frame += 1
    start = time.time()
    # 5.2获得预测结果
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    if args.indoor: #使用yolov5本身权重
        labels, scores, boxes = detector.run(
            cur_frame1, conf_thres=0.50, classes=[0]
        ) # person
    else:
        if args.modality.lower() == 'rgb': #用RGB图像权重
            labels, scores, boxes = detector1.run(
                cur_frame1, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
            
        elif args.modality.lower() == 't':  #用T图像权重
            labels, scores, boxes = detector2.run(
                cur_frame2, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
        elif args.modality.lower() == 'dual':  #用T图像权重
            labels, scores, boxes = detector3.run(
                cur_frame1,cur_frame2, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
        elif args.modality.lower() == 'rgbt': #双模态都用
            #获取RGB的预测结果    类别、置信分数、检测框
            labels1, scores1, boxes1 = detector1.run(
                cur_frame1, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
            #print("rgb",labels1, scores1, boxes1)
            #获取T的预测结果
            labels2, scores2, boxes2 = detector2.run(
                cur_frame2, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
            #print("T",labels2, scores2, boxes2)
            # 确定最终结果
            labels = labels1 + labels2 #合并类别数组
            #print("labels",labels)
            scores = scores1 + scores2 #合并分数数组
            #print("scores",scores)
            if boxes1.shape[0] > 0 and boxes2.shape[0] > 0: #如果可见光和红外都检测到目标
                boxes = np.concatenate([boxes1, boxes2], axis=0) #链接两个检测框
                #print("boxes",boxes)
                # 排除重复的目标框
                indices = simplified_nms(boxes, scores)
                labels, scores, boxes = np.array(labels)[indices], np.array(scores)[indices], boxes[indices]
                #print("result",labels, scores, boxes)
            elif boxes1.shape[0] > 0: #如果只有可见光检测到
                boxes = boxes1
                #print("boxes",boxes)
            elif boxes2.shape[0] > 0: #如果只有红外检测到
                boxes = boxes2
                #print("boxes",boxes)
            else:   #都没检测到
                boxes = np.array([])
                #print("boxes",boxes)
        else:
            raise ValueError("The modality must be 'RGB', 'T','dual' or 'RGBT'.")
    labels_temp = labels.copy()
    labels = []
    for i in labels_temp:
        labels.append(i if i not in ['pedestrian', 'cyclist'] else 'person')
    # 5.3单目估计距离
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    locations = mono.estimate(boxes)#获得目标框的世界坐标系
    indices = [i for i in range(len(locations)) if locations[i][1] > 0 and locations[i][1] < 200]
    labels, scores, boxes, locations = \
        np.array(labels)[indices], np.array(scores)[indices], boxes[indices], np.array(locations)[indices]
    distances = [(loc[0] ** 2 + loc[1] ** 2) ** 0.5 for loc in locations] #估计距离
    cur_frame1 = cur_frame1[:, :, ::-1].copy() # to BGR
    cur_frame2 = cur_frame2[:, :, ::-1].copy() # to BGR
    # 5.4画检测框
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    for i in reversed(np.argsort(distances)):#reversed()函数用于对可迭代对象中的元素进行反向排列
        cur_frame1 = draw_predictions(
            cur_frame1, str(labels[i]), float(scores[i]), boxes[i], location=locations[i] #两个图像上的数据显示一样
        )
        cur_frame2 = draw_predictions(
            cur_frame2, str(labels[i]), float(scores[i]), boxes[i], location=locations[i]
        )
    # 5.5发布图像
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    result_frame = np.concatenate([cur_frame1, cur_frame2], axis=1) #合并图像
    
    if args.display:
        if not display(result_frame, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")
    result_frame = result_frame[:, :, ::-1] # to RGB
    publish_image(pub, result_frame)
    delay = round(time.time() - start, 3)
    
    if args.print:
        print_info(frame, cur_stamp1, delay, labels, scores, boxes, locations, file_name)

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("dual_modal_perception", anonymous=True, disable_signals=True)
    frame = 0
    
    # 一、加载标定参数
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    if not os.path.exists(args.calib_file):
        raise ValueError("%s Not Found" % (args.calib_file))
    mono = MonoEstimator(args.calib_file, print_info=args.print)
    
    # 二、初始化Yolov5Detector
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    if args.indoor:
        detector = Yolov5Detector(weights='weights/coco/yolov5s.pt')
    else:
        if args.modality.lower() == 'rgb':
            detector1 = Yolov5Detector(weights='weights/seumm_visible/yolov5s_100ep_pretrained.pt')
        elif args.modality.lower() == 't':
            detector2 = Yolov5Detector(weights='weights/seumm_lwir/yolov5s_100ep_pretrained.pt')
        elif args.modality.lower() == 'dual':
            detector3 = Yolov5Detector(weights='weights/dual1.pt')
        elif args.modality.lower() == 'rgbt': #双模态
            detector1 = Yolov5Detector(weights='weights/10000myvis.pt')
            detector2 = Yolov5Detector(weights='weights/20000lrwyolov5.pt')
        else:
            raise ValueError("The modality must be 'RGB', 'T' or 'RGBT'.")
    # 三、进入回调函数
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # 准备图像序列
    image1_stamp, image1_frame = None, None
    image2_stamp, image2_frame = None, None
    
    #rgb回调函数
    rospy.Subscriber(args.sub_image1, Image, image1_callback, queue_size=1,
        buff_size=52428800)
    #红外回调函数
    rospy.Subscriber(args.sub_image2, Image, image2_callback, queue_size=1,
        buff_size=52428800)
    # 等待RGB和t图像都获得再进行下一次循环
    while image1_frame is None or image2_frame is None:
        time.sleep(0.1)
        print('Waiting for topic %s and %s...' % (args.sub_image1, args.sub_image2))
    print('  Done.\n')
    
    # 四、功能选择
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    #如果在命令行中输入 python3 demo_dual_modal.py --print 则会运行下面代码
    # 是否记录时间戳和检测结果
    if args.print:
        file_name = 'result.txt'
        with open(file_name, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    # 是否保存视频
    if args.display:
        assert image1_frame.shape == image2_frame.shape, \
            'image1_frame.shape must be equal to image2_frame.shape.'
        win_h, win_w = image1_frame.shape[0], image1_frame.shape[1] * 2
        v_path = 'result.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_writer = cv2.VideoWriter(v_path, v_format, args.frame_rate, (win_w, win_h), True)
    
    # 五、预测结果与发布
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # 启动定时检测线程
    pub = rospy.Publisher(args.pub_image, Image, queue_size=1)
    rospy.Timer(rospy.Duration(1 / args.frame_rate), timer_callback) #每frame_rate 秒调用一次timer_callback
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
    
