# -*- coding: utf-8 -*-
"""
该脚本用于对指定的视频文件进行YOLOv8目标检测。

它会加载一个预训练的YOLO模型和一个视频文件，
然后逐帧处理视频，将检测结果（边界框和标签）绘制在每一帧上，
并实时显示处理后的视频流。用户可以按 'q' 键关闭视频窗口。
"""
import cv2
import os
from ultralytics import YOLO

def detect_video_and_show():
    """
    加载视频和YOLO模型，对视频进行逐帧检测，并实时显示结果。
    """
    # --- 1. 配置路径 ---
    # 请根据您的文件位置修改下面的模型路径和视频路径
    
    # 模型权重文件的路径
    MODEL_PATH = 'runs/train_yolo12n/weights/best.pt'
    
    # 需要进行检测的视频的路径
    VIDEO_PATH = 'test_media/test_video.mp4' 

    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件未找到，路径: '{MODEL_PATH}'")
        return
        
    if not os.path.exists(VIDEO_PATH):
        print(f"错误：视频文件未找到，路径: '{VIDEO_PATH}'")
        return

    # --- 3. 加载模型和视频 ---
    try:
        # 加载YOLO模型
        print(f"正在加载模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        
        # 打开视频文件
        print(f"正在加载视频: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件: '{VIDEO_PATH}'")
            return
            
    except Exception as e:
        print(f"加载模型或视频时发生错误: {e}")
        return

    # --- 4. 逐帧处理视频 ---
    print("开始逐帧检测视频... 按 'q' 键退出。")
    
    while cap.isOpened():
        # 从视频中读取一帧
        success, frame = cap.read()
        
        if success:
            # 对当前帧进行YOLOv8推理
            results = model(frame)
            
            # 在帧上可视化结果
            annotated_frame = results[0].plot()
            
            # 显示带标注的帧
            cv2.imshow("YOLO Video Detection", annotated_frame)
            
            # 如果按下 'q' 键，则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果视频结束，则退出循环
            print("视频播放完毕。")
            break
            
    # --- 5. 释放资源 ---
    # 释放视频捕获对象
    cap.release()
    # 销毁所有OpenCV创建的窗口
    cv2.destroyAllWindows()
    print("窗口已关闭，程序结束。")


if __name__ == "__main__":
    detect_video_and_show()