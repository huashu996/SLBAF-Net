#!/usr/bin/env python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse #命令行解析模块
import os
import platform
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from pathlib import Path
from thop import profile
from thop import clever_format
import torch
import torch.backends.cudnn as cudnn
from ptflops import get_model_complexity_info
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/test_images',  # file/dir/URL/glob, 0 for webcam
        source2=ROOT / 'data/test_images2',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        project2=ROOT / 'runs/detect2',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)#是否要用电脑摄像头
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir2 = increment_path(Path(project2) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir2 / 'labels' if save_txt else save_dir2).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device) #指定设备
    #加载模型
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    print(model.names)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸，确保能被32整除

    # Dataloader 加载数据
    if webcam: #电脑摄像头
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else: #数据集
        dataset = LoadImages(source,source2, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    #model.warmup(imgsz=(1 if pt else bs,imgsz2=(1 if pt else bs, 3, *imgsz))  # warmup 模型预热
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    '''
    path 图片视频路径
    img  进行resize+pad之后的图片,如（3,640,512)（c,h,w）
    img0s 原size图片，（1080,810,3）
    cap 读取图片时为None 读取视频时为视频源
    '''
    for path,path2, im, im0s,im2,im0s2, vid_cap, s in dataset:
        t1 = time_sync() #获取时间
        im = torch.from_numpy(im).to(device) #转化为tensor格式
        im2 = torch.from_numpy(im2).to(device) #转化为tensor格式
        im =  im.float()  # uint8 to fp16/32
        im2 = im2.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0 #0～1中间的值
        im2 /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
            im2 = im2[None] 
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        visualize2 = increment_path(save_dir2 / Path(path2).stem, mkdir=True) if visualize else False

        pred = model(im,im2, augment=False, visualize=visualize) #将图片传入模型网络
        macs, params = profile(model, inputs=(im,im2), verbose = False)
        print(f"macs = {macs/1e9}G")
        print(f"params = {params/1e6}M")
        #cam热力图

        
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        # pred :前向传播的输出
        # conf_thres 置信度阈值
        # classes 是否保留特定类别
        # 经过nms之后，预测框格式会变从 xywh变成xyxy
        pred1 = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred2 = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 对每一张图片处理 i表示第几个框

        get_box(pred1,seen,path,im,im0s,dataset,save_dir,save_crop,save_txt,names,save_img,hide_labels,hide_conf,view_img,s,dt,imgsz)
        get_box(pred2,seen,path2,im2,im0s2,dataset,save_dir2,save_crop,save_txt,names,save_img,hide_labels,hide_conf,view_img,s,dt,imgsz)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def get_box(pred,seen,path,im,im0s,dataset,save_dir,save_crop,save_txt,names,save_img,hide_labels,hide_conf,view_img,s,dt,imgsz):
    for i, det in enumerate(pred):  # per image
        seen += 1
        s += '%gx%g ' % im.shape[2:]  # 设置打印信息图片宽高
        # Print time (inference-only)
        
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # 保存图片路径
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 设置保存框坐标的txt文件
        
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # 调整预测框的坐标，基于resize+pad的图片坐标转化为原size图像上的坐标
            #此时坐标格式是xyxy
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            #print(det[:, :4])
            # Print results 打印检测到结果的类别数目
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results 保存预测结果
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 将xyxy格式转化为xywh，并除上wh做归一化，转为列表保存
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                #在原图上画框
                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        # Stream results
        im0 = annotator.result()
        if view_img: #显示预测图片
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img: #保存预测后的图片
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

        
def parse_opt():
    #建立参数解析对象parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/test_images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source2', type=str, default=ROOT / 'data/test_images2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w') #网络输入图片的大小
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold') #置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')#iou阈值
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') #设置设备
    parser.add_argument('--view-img', action='store_true', help='show results') #是否展示预测之后的视频
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')#是否将预测的框以txt保存
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')#是否将置信度保存
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3') #设置只保留某一部分类别
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')#进行nms是否去除不同类别之间的框
    parser.add_argument('--augment', action='store_true', help='augmented inference')#推理时候进行多尺度翻转
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models') #对所有模型进行strip_optimizer操作
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference') #是否是半精度
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args() #参数都会放到opt
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    
    opt = parse_opt()
    main(opt)
