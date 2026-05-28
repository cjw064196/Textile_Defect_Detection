# -*- coding: utf-8 -*-
"""
该脚本用于对单张静态图片进行YOLOv8目标检测。

它会加载一个预训练的YOLO模型和一张指定的图片，
执行检测后，会同时在两个窗口中分别显示原始图片和带有检测结果的图片。
用户按任意键后关闭窗口。
"""
import cv2
import os
from ultralytics import YOLO
import numpy as np

def cv_imread(file_path):
    """
    使用OpenCV读取可能包含中文等非ASCII字符路径的图片。

    Args:
        file_path (str): 图片文件的路径。

    Returns:
        numpy.ndarray: 读取到的图像数据 (BGR格式)。
    """
    # 使用numpy从文件读取数据，然后用imdecode解析，可以避免中文路径问题
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

def detect_and_show():
    """
    加载图像和YOLO模型，执行检测，并显示原始图和结果图。
    """
    # --- 1. 配置路径 ---
    # 请根据您的文件位置修改下面的模型路径和图片路径
    
    # 模型权重文件的路径
    MODEL_PATH = 'runs/train_yolo12n/weights/best.pt'
    
    # 需要进行检测的图片的路径
    # 请确保这里指向一个真实存在的图片文件
    IMAGE_PATH = 'test_media/delik-115-_jpg.rf.883358bd77fe7a9bd93c8001aefb9fc9.jpg' 

    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件未找到，路径: '{MODEL_PATH}'")
        return
        
    if not os.path.exists(IMAGE_PATH):
        print(f"错误：图片文件未找到，路径: '{IMAGE_PATH}'")
        print("请检查脚本中的 IMAGE_PATH 变量是否指向一个有效的图片。")
        return

    # --- 3. 加载模型和图片 ---
    try:
        # 加载YOLOv8模型
        print(f"正在加载模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        
        # 使用能处理特殊字符路径的函数加载原始图片
        print(f"正在加载图片: {IMAGE_PATH}")
        original_image = cv_imread(IMAGE_PATH)
        
        if original_image is None:
            print(f"错误：无法从路径加载图片: '{IMAGE_PATH}'")
            return
            
    except Exception as e:
        print(f"加载模型或图片时发生错误: {e}")
        return

    # --- 4. 执行目标检测 ---
    try:
        print("正在执行目标检测...")
        # 对图片进行推理
        results = model.predict(source=original_image)
        
        # 从结果中获取带标注的图片
        # results[0].plot() 会返回一个带有检测框和标签的图像 (BGR格式)
        result_image = results[0].plot()
        print("检测完成。")
        
    except Exception as e:
        print(f"检测过程中发生错误: {e}")
        return

    # --- 5. 显示图片 ---
    # 在窗口中显示原始图片
    cv2.imshow('Original Image', original_image)
    
    # 在另一个窗口中显示检测结果图片
    cv2.imshow('Detection Result', result_image)
    
    # 等待用户按键后关闭所有窗口
    print("检测结果已显示。请在图片窗口激活的状态下按任意键关闭...")
    cv2.waitKey(0)
    
    # 销毁所有OpenCV创建的窗口
    cv2.destroyAllWindows()
    print("窗口已关闭。")

if __name__ == "__main__":
    detect_and_show()