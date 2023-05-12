# dual_modal_perception
论文名称《SLBAF-Net: Super-Lightweight bimodal adaptive fusion network for UAV detection in low recognition environment》
论文地址  https://link.springer.com/article/10.1007/s11042-023-15333-w
ROS package for dual modal perception (rgbt)
## 简介
   此双模态检测是对可见光和红外图像进行卷积融合，针对于无人机等小目标的检测。
## 安装
 - 建立ROS工作空间并拷贝这个库
   ```Shell
   mkdir -p ros_ws/src
   cd ros_ws/src
   git clone git@github.com:huashu996/dual_conv_fusion_yolov5.git --recursive
   ```
 - 使用Anaconda设置环境依赖
   ```Shell
   conda create -n yolov5.v5.0 python=3.8
   conda activate yolov5.v5.0
   cd dual_modal_perception
   pip install -r requirements.txt
   pip install catkin_tools
   pip install rospkg
   ```
## 训练测试
 - 制作双模态数据集txt形式
   将数据集images images2 labels 放入data文件夹下，注意名字一样，最后执行命令
   ```Shell  
   python split_train_val.py 
   python voc_label.py 
   python voc_label2.py 
   ```
   即可看到生成train.txt val.txt test.txt train2.txt val2.txt test2.txt 文件
   随后修改双模态的yaml文件
   ```Shell  
    train: data/train.txt
	train2: data/train2.txt
	val: data/val.txt
	val2: data/val2.txt
	test: data/test.txt
	# Classes
	nc: 7  # number of classes
	names: ['pedestrian','cyclist','car','bus','truck','traffic_light','traffic_sign']  # class names
   ```
 - 训练
   ```Shell  
   python train.py --data data/dual.yaml --cfg models/attentionuav7.yaml --weights weights/yolov5s.pt --batch-size 2 --epochs 50
   ```
 - 测试
   ```Shell  
   python detect.py --weights ./weights/dual.pt --source data/test_images --source2 data/test_images2
   ```
 - 验证
   python val.py --weights ./weights/dual.pt --data ./data/dual.yaml 

## ROS环境跑
 - 编写相机标定参数`dual_modal_perception/conf/calibration_image.yaml`
   ```
   %YAML:1.0
   ---
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [859, 0, 339, 0, 0, 864, 212, 0, 0, 0, 1, 0]
   Height: 2.0 # the height of the camera (meter)
   DepressionAngle: 0.0 # the depression angle of the camera (degree)
   ```
 - 更换权重
   改动demo_dual_modal.py文件中权重的路径  
   ```Shell  
        elif args.modality.lower() == 'dual':
            detector3 = Yolov5Detector(weights='weights/dual1.pt')
   ```

 - 启动双模态检测算法（检测结果图像可由`rqt_image_view /result`查看）
   ```
   python3 demo_dual_modal.py
   
   # If you want print infos and save videos, run
   python3 demo_dual_modal.py --print --display
   ```

