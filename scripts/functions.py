import cv2
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image

def get_stamp(header): #计算时间戳
    return header.stamp.secs + 0.000000001 * header.stamp.nsecs

def publish_image(pub, data, frame_id='base_link'): #发布消息
    assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
    header = Header(stamp=rospy.Time.now())
    header.frame_id = frame_id
    
    msg = Image()
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = 'rgb8'
    msg.data = np.array(data).tostring()
    msg.header = header
    msg.step = msg.width * 1 * 3
    
    pub.publish(msg)

def display(img, v_writer, win_name='result'): #保存视频
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    v_writer.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        v_writer.release()
        return False
    else:
        return True

def print_info(frame, stamp, delay, labels, scores, boxes, locs, file_name='result.txt'):
    time_str = 'frame:%d  stamp:%.3f  delay:%.3f' % (frame, stamp, delay)
    print(time_str)
    with open(file_name, 'a') as fob:
        fob.write(time_str + '\n')
    for i in range(len(labels)):
        info_str = 'box:%d %d %d %d  loc:(%.2f, %.2f)  score:%.2f  label:%s' % (
            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], locs[i][0], locs[i][1], \
            scores[i], labels[i]
        )
        print(info_str)
        with open(file_name, 'a') as fob:
            fob.write(info_str + '\n')
    print()
    with open(file_name, 'a') as fob:
        fob.write('\n')


def simplified_nms(boxes, scores, iou_thres=0.5):  #排除重复目标
    '''
    Args:
        boxes: (n, 4), xyxy format
        scores: list(float)
    Returns:
        indices: list(int), indices to keep
    '''
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] #获得每个检测框的对角坐标
    area = (x2 - x1) * (y2 - y1) #所有检测框面积
    #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，使用[::-1],可以建立X从大到小的索引。
    #1、置信度排序
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    idx = np.argsort(scores)[::-1] #置信度由大到小的索引排序
    print("idx",idx)
    indices = []
    #2、循环数组筛选
    #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    while len(idx) > 0:
        i = idx[0] #取最大置信度的索引
        indices.append(i)  #将这个索引放入indices中
        #print('indices',indices)
        if len(idx) == 1:  #如果就一个检测框
            break
        idx = idx[1:] #意思是去掉列表中第一个元素，对后面的元素进行操作
        #print('idx[1:]',idx)
        #3、与最大置信度框的重合度来判断是否是重复检测框
        #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
        xx1 = np.clip(x1[idx], a_min=x1[i], a_max=np.inf) #截取数组中小于或者大于某值的部分 np.inf表示无穷大
        yy1 = np.clip(y1[idx], a_min=y1[i], a_max=np.inf)
        xx2 = np.clip(x2[idx], a_min=0, a_max=x2[i])
        yy2 = np.clip(y2[idx], a_min=0, a_max=y2[i])
        w, h = np.clip((xx2 - xx1), a_min=0, a_max=np.inf), np.clip((yy2 - yy1), a_min=0, a_max=np.inf)
        
        inter = w * h
        union = area[i] + area[idx] - inter
        iou = inter / union #计算iou 越大越好最大为1
        #print('iou',iou)
        #4、将重合度小的即不同目标保存到idx再次循环
        #×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
        idx = idx[iou < iou_thres] #框重合度小视为不同目标
        #print("idx",idx)
        
    return indices #返回合并后的索引
