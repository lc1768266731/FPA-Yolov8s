# FPA-Yolov8s（在master分支）
FPA-YOLOv8 是基于官方 YOLOv8 库编写的，用于无人机航拍图像中的目标检测。官方 YOLOv8 库的 URL 为：https://github.com/ultralytics/ultralytics/tree/main。

1.将模型文件夹中的 yaml 文件添加到 YOLOv8 官方库中。

2.在ultralytics-8.1.0/ultralytics/nn/modules文件夹中创建C2f_PPA.py，将上述代码复制到此

3.在ultralytics-8.1.0/ultralytics/nn/modules文件夹中创建ADM.py,将上述代码复制到此

4.之后运行train.py,val.py,进行训练，验证和测试
