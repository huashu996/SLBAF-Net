import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile
from PIL import Image
from PIL import ImageStat
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes  #格式支持的图片
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes #视频格式
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path,path2, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        p2 = str(Path(path2).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
            files2 = sorted(glob.glob(p2, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            files2 = sorted(glob.glob(os.path.join(p2, '*.*')))  # dir
        elif os.path.isfile(p): #如果是文件直接获取
            files = [p]  # files
            files2 = [p2]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        #分别提取图片和视频的路径
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        images2 = [x for x in files2 if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos) #获取数量

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos #整个图片视频放一个列表
        self.files2 = images2 + videos #整个图片视频放一个列表
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv#判断是否为视频，方便后续单独处理
        self.mode = 'image'
        self.auto = auto
        if any(videos): #是否包含视频文件
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self): #创建迭代器对象
        self.count = 0
        return self

    def __next__(self): #输出下一项
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        path2 = self.files2[self.count]

        if self.video_flag[self.count]: #如果为视频
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR格式
            img02 = cv2.imread(path2)  # BGR格式
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0] #对图片缩放填充
        img2 = letterbox(img02, self.img_size, stride=self.stride, auto=self.auto)[0] #对图片缩放填充

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB #BGR到RGB的转换
        img = np.ascontiguousarray(img) #将数组转换为连续，提高速度
        img2 = img2.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB #BGR到RGB的转换
        img2 = np.ascontiguousarray(img2) #将数组转换为连续，提高速度

        return path,path2, img, img0,img2, img02, self.cap, s

    def new_video(self, path):
        self.frame = 0 #frme记录帧数
        self.cap = cv2.VideoCapture(path) #初始化视频对象
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) #总帧数

    def __len__(self):
        return self.nf  # number of files
#返回文件列表的hash值
def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

#获取图片宽高
def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]#对图片进行旋转
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s
def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

def create_dataloader(path,path2,imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='',prefix2='', shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        #return torch.from_numpy(img),torch.from_numpy(img2), labels_out, self.im_files[index],self.im_files2[index], shapes
        dataset = LoadImagesAndLabels(path,path2, imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      prefix2=prefix2)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler) #返回训练集图片个数

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def cache_labels(path, im_files,label_files,prefix='',):
    cache_version = 0.6
    # Cache dataset labels, check images and read shapes
    x = {}  # dict
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
    with Pool(NUM_THREADS) as pool:
        pbar = tqdm(pool.imap(verify_image_label, zip(im_files, label_files, repeat(prefix))),
                    desc=desc, total=len(im_files), bar_format=BAR_FORMAT)
        for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if im_file:
                x[im_file] = [lb, shape, segments]# 保存为字典
            if msg:
                msgs.append(msg)
            pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

    pbar.close()
    if msgs:
        LOGGER.info('\n'.join(msgs))
    if nf == 0:
        LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
    x['hash'] = get_hash(label_files + im_files)
    x['results'] = nf, nm, ne, nc, len(im_files)
    x['msgs'] = msgs  # warnings
    x['version'] = cache_version  # cache version
    try:
        np.save(path, x)  # save cache for next time 保存本地方便下次使用
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {path}')
    except Exception as e:
        LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
    return x
def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    #图片角度旋转矫正
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        #标签过滤
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment 轮廓点
                    classes = np.array([x[0] for x in lb], dtype=np.float32) #第一个数是类别
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32) #保存边框数据
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                #归一化
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates #去除重复的数据
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]
            
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
def img2label_paths2(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images2' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
def get_cache(path,mode,prefix):
    prefix=prefix
    cache_version = 0.6  # dataset labels *.cache version
    try:
        #1、获取图片
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep #上级目录os.sep是分隔符
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise Exception(f'{prefix}{p} does not exist')
        # 2、过滤不支持格式的图片
        im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert im_files, f'{prefix}No images found'
    except Exception as e:
        raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')
    if mode==1:
        label_files = img2label_paths(im_files)  # 获取labels
    elif mode==2:
        label_files = img2label_paths2(im_files)  # 获取labels
    cache_path = (p if p.is_file() else Path(label_files[0]).parent).with_suffix('.cache')
    try:
        cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
        assert cache['version'] == cache_version  # same version
        assert cache['hash'] == get_hash(label_files + im_files)  # same hash 判断hash值是否改变
    except Exception:
        cache, exists = cache_labels(cache_path,im_files,label_files, prefix), False  # cache

    # Display cache  过滤结果打印
    nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
    if exists:
        d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
        tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
        if cache['msgs']:
            LOGGER.info('\n'.join(cache['msgs']))  # display warnings
    return cache
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#一、数据处理
class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    

    def __init__(self, path, path2, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='',prefix2=''):
        #创建参数
        self.img_size = img_size
        self.augment = augment #是否数据增强
        self.hyp = hyp #超参数
        self.image_weights = image_weights #图片采样权重
        self.rect = False if image_weights else rect #矩阵训练
        #mosaic数据增强
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride #最大下采样步数
        self.path = path
        self.path2 = path2
        self.albumentations = Albumentations() if augment else None
        self.prefix=prefix
        self.prefix2=prefix2
        cache = get_cache(self.path,1,self.prefix)
        cache2 = get_cache(self.path2,2,self.prefix2)
        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update 图片列表
        self.label_files = img2label_paths(cache.keys())  # update 标签列表
        n = len(shapes)  # number of images 14329
        bi = np.floor(np.arange(n) / batch_size).astype(np.int_)  # batch index 将每一张图片batch索引
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)
        
        [cache2.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels2, shapes2, self.segments2 = zip(*cache2.values())
        self.labels2 = list(labels2)
        self.shapes2 = np.array(shapes2, dtype=np.float64)
        self.im_files2 = list(cache2.keys())  # update 图片列表
        self.label_files2 = img2label_paths(cache2.keys())  # update 标签列表
        n2 = len(shapes2)  # number of images 14329
        bi2 = np.floor(np.arange(n2) / batch_size).astype(np.int_)  # batch index 将每一张图片batch索引
        nb2 = bi2[-1] + 1  # number of batches
        self.batch2 = bi2  # batch index of image
        self.n2 = n2
        self.indices2 = range(n2)
        

        # Update labels
        #过滤类别
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0 把所有目标归为一类
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0
        include_class2 = []  # filter labels to include only these classes (optional)
        include_class_array2 = np.array(include_class2).reshape(1, -1)
        for i, (label2, segment2) in enumerate(zip(self.labels2, self.segments2)):
            if include_class2:
                j = (label2[:, 0:1] == include_class_array2).any(1)
                self.labels2[i] = label2[j]
                if segment2:
                    self.segments2[i] = segment2[j]
            if single_cls:  # single-class training, merge all classes into 0 把所有目标归为一类
                self.labels2[i][:, 0] = 0
                if segment2:
                    self.segments2[i][:, 0] = 0
        #是否采用矩形构造
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio #高和宽的比
            irect = ar.argsort() #根据ar排序
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            
            # Set training image shapes 设置训练图片的shapes
            # 对同个batch进行尺寸处理
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int_) * stride
            

            
            s2 = self.shapes2  # wh
            ar2 = s2[:, 1] / s2[:, 0]  # aspect ratio #高和宽的比
            irect2 = ar2.argsort() #根据ar排序
            self.im_files2 = [self.im_files2[i] for i in irect2]
            self.label_files2 = [self.label_files2[i] for i in irect2]
            self.labels2 = [self.labels2[i] for i in irect2]
            self.shapes2 = s2[irect2]  # wh
            ar2 = ar2[irect2]
            
            shapes2 = [[1, 1]] * nb2
            for i in range(nb2):
                ari2 = ar2[bi2 == i]
                mini, maxi = ari2.min(), ari2.max()
                if maxi < 1:
                    shapes2[i] = [maxi, 1]
                elif mini > 1:
                    shapes2[i] = [1, 1 / mini]
            self.batch_shapes2 = np.ceil(np.array(shapes2) * img_size / stride + pad).astype(np.int_) * stride

            
        self.ims = [None] * self.n
        self.ims2 = [None] * self.n2
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        self.npy_files2 = [Path(f).with_suffix('.npy') for f in self.im_files2]
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    def __len__(self):
        return len(self.im_files)
        #加载图片并根据设定输入大小与图片源大小比例进行resize
    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i], #判断有没有这个图片
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            #根据r选择不同的插值
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im,
                                (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized
    def load_image2(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims2[i], self.im_files2[i], self.npy_files2[i], #判断有没有这个图片
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            #根据r选择不同的插值
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im,
                                (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized
    #二、图片增强
    def __getitem__(self, index):#根据每个类别数量获得图片采样权重，获取新的下标
        i = self.indices[index]  # linear, shuffled, or image_weights
        i2 = self.indices2[index]  # linear, shuffled, or image_weights
        hyp = self.hyp
         #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        if mosaic:
            # Load mosaic
            img,img2, labels = self.load_mosaic9(index) #mosaic数据增强的方式加载图片标签
            shapes = None
            shapes2 = None
            #是否做Mixup数据增强
            # MixUp augmentation
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            img2, (h02, w02), (h2, w2) = self.load_image2(index)
            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            labels = self.labels[index].copy()
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            
            shape2 = self.batch_shapes2[self.batch2[index]] if self.rect else self.img_size  # final letterboxed shape
            img2, ratio2, pad2 = letterbox(img2, shape2, auto=False, scaleup=self.augment)
            labels2 = self.labels2[index].copy()
            shapes2 = (h02, w02), ((h2 / h02, w2 / w02), pad2)  # for COCO mAP rescaling
            
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            if labels2.size:  # normalized xywh to pixel xyxy format
                labels2[:, 1:] = xywhn2xyxy(labels2[:, 1:], ratio2[0] * w2, ratio2[1] * h2, padw=pad2[0], padh=pad2[1])
            #大小缩放
            #——————————————————————————————————————————————————————————————————————————————————————————————————
            if self.augment: 
                img,img2, labels = random_perspective(img,img2, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
        
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
        #翻转色调
        #——————————————————————————————————————————————————————————————————————————————————————————————————
        if self.augment: 
            # Albumentations
            #进一步数据增强
            img,img2, labels = self.albumentations(img,img2, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            augment_hsv(img2, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                img2 = np.flipud(img2)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img) #沿轴 1(左/右)反转元素的顺序。
                img2 = np.fliplr(img2)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout
        
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
            
            

        '''

       load_mosaic9
        print(weight)
        print("____________________")
        print(img_1.size)
        cv2.imshow('gray',img_1)
        cv2.waitKey(0)
        '''
        
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img2 = img2.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img2 = np.ascontiguousarray(img2)
        return torch.from_numpy(img),torch.from_numpy(img2), labels_out, self.im_files[index],self.im_files2[index], shapes,shapes2
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    def mean_pooling(self,img, G=4):
        # Max Pooling
        out = img.copy()
        H, W = img.shape
        Nh = int(H / G)
        Nw = int(W / G)
        for y in range(Nh):
            for x in range(Nw):
                    out[G*y:G*(y+1), G*x:G*(x+1)] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1)])
        return out
        
    def load_mosaic(self, index): #self自定义数据集 index要增强的索引
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        #随机选取一个中心点
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        #随机取其他三张图片索引
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)#load_image 加载图片根据设定的输入大小与图片原大小的比例进行resize
            img2, _, (h2, w2) = self.load_image2(index)#load_image 加载图片根据设定的输入大小与图片原大小的比例进行resize

            # place img in img4
            if i == 0:  # top left
                #初始化大图
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                img42 = np.full((s * 2, s * 2, img2.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                #把原图放到左上角
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                #选取小图上的位置 如果图片越界会裁剪
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            #小图上截取的部分贴到大图上
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            img42[y1a:y2a, x1a:x2a] = img2[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            # 计算小图到大图后的偏移 用来确定目标框的位置
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            #标签裁剪
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels) #得到新的label的坐标
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # 将图片中没目标的 取别的图进行粘贴
        img4,img42, labels4, segments4 = copy_paste(img4,img42,labels4, segments4, p=self.hyp['copy_paste'])
        # 随机变换
        img4,img42,labels4 = random_perspective(img4,img42,labels4, segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                          perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img4,img42, labels4 #返回数据增强的后的图片和标签
    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)
            img_1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            means, dev = cv2.meanStdDev(img_1)
            weight = 1*np.exp(-((means-127.5)/41.07)**2)
            img=img*weight
            img2, _, (h2, w2) = self.load_image2(index)#load_image 加载图片根据设定的输入大小与图片原大小的比例进行resize
            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                img92 = np.full((s * 3, s * 3, img2.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            img92[y1:y2, x1:x2] = img2[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]
        img92 = img92[yc:yc + 2 * s, xc:xc + 2 * s]
        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9,img92, labels9 = random_perspective(img9,img92, labels9, segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9,img92, labels9

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    @staticmethod
    def collate_fn(batch): #如何取样本
        im,im2, label, path,path2, shapes,shapes2 = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0),torch.stack(im2, 0), torch.cat(label, 0), path,path2, shapes,shapes2
    @staticmethod
    def collate_fn4(batch):
        im,im2, label, path,path2, shapes,shapes2 = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4,im42, label4, path4,path42, shapes4,shapes42 = [],[], [], path[:n],path2[:n2],shapes[:n],shapes2[:n2]
        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                im2 = F.interpolate(img2[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                lb = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                im2 = torch.cat((torch.cat((img2[i], img2[i + 1]), 1), torch.cat((img2[i + 2], img2[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im)
            im42.append(im2)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im4, 0),torch.stack(im42, 0), torch.cat(label4, 0), path4,path42,shapes4
