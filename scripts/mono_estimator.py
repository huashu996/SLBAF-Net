# -*- coding: UTF-8 -*- 

import numpy as np
import math
import cv2
from math import sin, cos

class MonoEstimator():
    def __init__(self, file_path, print_info=True):
        fs = cv2.FileStorage(file_path, cv2.FileStorage_READ)
        
        mat = fs.getNode('ProjectionMat').mat()
        self.fx = int(mat[0, 0])
        self.fy = int(mat[1, 1])
        self.u0 = int(mat[0, 2])
        self.v0 = int(mat[1, 2])
        
        self.height = fs.getNode('Height').real()
        self.depression = fs.getNode('DepressionAngle').real() * math.pi / 180.0
        
        if print_info:
            print('Calibration of camera:')
            print('  Parameters: fx(%d) fy(%d) u0(%d) v0(%d)' % (self.fx, self.fy, self.u0, self.v0))
            print('  Height: %.2fm' % self.height)
            print('  DepressionAngle: %.2frad' % self.depression)
            print()
    
    def uv_to_xyz(self, u, v):
        # Compute (x, y, z) coordinates in the real world, according to (u, v) coordinates in the image.
        # X axis - on the right side of the camera
        # Z axis - in front of the camera
        u = int(u)
        v = int(v)
        
        fx, fy = self.fx, self.fy
        u0, v0 = self.u0, self.v0
        
        h = self.height
        t = self.depression
        
        denominator = fy * sin(t) + (v - v0) * cos(t)
        if denominator != 0:
            z = (h * fy * cos(t) - h * (v - v0) * sin(t)) / denominator
            if z > 1000: z = 1000
        else:
            z = 1000
        x = (z * (u - u0) * cos(t) + h * (u - u0) * sin(t)) / fx
        y = h
        return x, y, z
    
    def estimate(self, boxes):#单目估计距离
        locations = []
        if boxes.shape[0] > 0: #判断目标个数是否大于1
            for box in boxes:
                print(box[0])
                u, v = (box[0] + box[2]) / 2, box[3]
                x, y, z = self.uv_to_xyz(u, v)
                locations.append((x, z))
        return locations
