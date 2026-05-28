# -*- coding: utf-8 -*-
"""
该脚本用于启动摄像头，进行实时的YOLO目标检测，并在窗口中显示结果。
用户可以通过按 'q' 键来关闭显示窗口并退出程序。
"""
import cv2
from ultralytics import YOLO
import os

# --- 配置 ---
# 模型路径 (请确保路径正确)
MODEL_PATH = 'runs/train_yolo12n/weights/best.pt'
# 摄像头索引 (0通常是默认摄像头)
CAMERA_INDEX = 0
# 窗口标题
WINDOW_TITLE = "YOLO 摄像头实时检测 (按 'q' 退出)"

def main():
    """
    主函数，用于启动摄像头并进行实时目标检测。
    """
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件未找到，路径: '{MODEL_PATH}'")
        print("请检查脚本中的 MODEL_PATH 变量是否指向一个有效的模型文件。")
        return

    print(f"正在加载模型: {MODEL_PATH}")
    # 加载YOLOv8模型
    model = YOLO(MODEL_PATH)

    # 打开摄像头
    print(f"正在打开摄像头，索引: {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开摄像头，索引: {CAMERA_INDEX}")
        print("请确保摄像头已连接并正常工作，或尝试更改 CAMERA_INDEX。")
        return

    print("开始实时检测... 按 'q' 键退出。")

    # 循环读取摄像头画面
    while True:
        # 读取一帧
        success, frame = cap.read()

        if success:
            # 对当前帧进行目标检测
            results = model(frame)

            # 将检测结果绘制在画面上
            annotated_frame = results[0].plot()

            # 显示处理后的画面
            cv2.imshow(WINDOW_TITLE, annotated_frame)

            # 检测按键，如果按下'q'键则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("检测到 'q' 键，正在退出...")
                break
        else:
            # 如果读取失败（例如摄像头断开），则退出循环
            print("无法从摄像头读取画面，程序结束。")
            break

    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已释放，窗口已关闭。")

if __name__ == "__main__":
    main()