import cv2
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Union, Any
import argparse
import threading
import torch

# 添加本地ultralytics目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ultralytics'))

from config_text import CLASS_NAME_MAP

# 全局模型缓存字典（格式: {模型路径: 模型实例}）
_model_cache = {}
_cache_lock = threading.Lock()  # 线程安全锁

def get_device():
    """获取当前可用的计算设备"""
    if torch.cuda.is_available():
        return '0' # 使用第一块GPU
    return 'cpu'

def load_model_with_cache(model_path: str, conf_threshold: float = 0.5) -> Any:
    """带缓存的模型加载函数"""
    # Lazy import to speed up server start
    from ultralytics import YOLO
    
    global _model_cache
    
    with _cache_lock:
        # 检查模型是否已缓存
        if model_path in _model_cache:
            model = _model_cache[model_path]
            model.conf = conf_threshold  # 更新置信度阈值
            return model
        
        # 加载新模型并缓存
        print(f"⏳ 正在加载模型: {model_path}")
        model = YOLO(model_path)
        model.conf = conf_threshold
        _model_cache[model_path] = model
        return model

def yolo_detection(
    image_path: str,
    model_path: str,
    output_image_path: Union[str, None] = None,
    output_json_path: Union[str, None] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    filtered_categories: Union[list, None] = None
) -> tuple:
    """
    YOLOv8目标检测函数
    
    参数:
        image_path (str): 输入图片路径
        model_path (str): 训练好的模型路径(.pt)
        output_image_path (str, optional): 输出图片路径，默认同输入路径加'_detected'
        output_json_path (str, optional): 输出JSON路径，默认同输入路径加'_result.json'
        conf_threshold (float): 置信度阈值，默认0.5
        iou_threshold (float): IOU阈值，默认0.5
        filtered_categories (list, optional): 需要保留的类别列表，默认None表示保留所有类别
    
    返回:
        tuple: (输出图片路径, 输出JSON路径)
    """

    # 从配置文件导入的中英文类别映射和相关建议映射

    # 处理默认输出路径
    image_path = str(Path(image_path).resolve())
    input_path = Path(image_path)
    
    if output_image_path is None:
        output_image_path = str(input_path.with_name(f"{input_path.stem}_detected{input_path.suffix}"))
    if output_json_path is None:
        output_json_path = str(input_path.with_name(f"{input_path.stem}_result.json"))

    # 加载模型
    model = load_model_with_cache(model_path, conf_threshold)

    # 读取并预处理图片
    if not Path(image_path).exists():
        raise FileNotFoundError(f"输入图片不存在: {image_path}")
    
    # Use imdecode to support non-ASCII paths on Windows
    # img = cv2.imread(image_path)
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading image with imdecode: {e}")
        img = None

    if img is None:
        # Fallback to standard imread if imdecode fails (though imdecode is usually more robust)
        img = cv2.imread(image_path)
        
    if img is None:
        raise ValueError(f"无法读取图片文件: {image_path}")

    # 推理
    import time
    device = get_device()
    print(f"🚀 Inference using device: {device}")
    start_time = time.time()
    results = model.predict(source=img, conf=conf_threshold, iou=iou_threshold, device=device)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 # 转换为毫秒

    # 解析结果
    output_data = {
        "detections": [],
        "inference_time": f"{inference_time:.1f}ms"
    }
    class_counter = {}
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = result.names[cls_id]
            print(f"检测到: {class_name}，置信度: {conf:.2f},id={cls_id}")
            chinese_name = CLASS_NAME_MAP.get(class_name, class_name)
            
            # 转换为xyxy坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 记录检测结果
            detection = {
                "class": chinese_name,
                "class_en": class_name,
                "cls_id": cls_id,
                "confidence": round(conf, 4),
                "bbox": {
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2
                }
            }
            output_data["detections"].append(detection)
            
            # 统计类别数量
            if class_name in class_counter:
                class_counter[class_name] += 1
            else:
                class_counter[class_name] = 1


    if class_counter:
        # 添加类别统计
        output_data["category_counts"] = {
            CLASS_NAME_MAP.get(k, k): v 
            for k, v in class_counter.items()
        }
    print("output_json",output_data)
    # 保存JSON
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    # 绘制检测框
    img_with_boxes = img.copy()
    
    # 预定义颜色调色板 (BGR格式)
    # 包含20种不同颜色，用于区分不同类别
    COLOR_PALETTE = [
        (51, 87, 255),   # 0: 红橙色
        (87, 255, 51),   # 1: 亮绿色
        (255, 87, 51),   # 2: 蓝色
        (255, 51, 243),  # 3: 紫粉色
        (168, 51, 255),  # 4: 粉紫色
        (246, 255, 51),  # 5: 青色
        (51, 214, 255),  # 6: 金黄色
        (255, 51, 168),  # 7: 紫罗兰
        (51, 51, 255),   # 8: 纯红
        (0, 255, 0),     # 9: 纯绿
        (255, 0, 0),     # 10: 纯蓝
        (0, 255, 255),   # 11: 黄色
        (255, 0, 255),   # 12: 品红
        (255, 255, 0),   # 13: 青色
        (0, 0, 255),     # 14: 深红
        (0, 128, 255),   # 15: 橙色
        (128, 0, 128),   # 16: 紫色
        (128, 128, 0),   # 17: 深青色
        (0, 128, 128),   # 18: 橄榄色
        (128, 0, 0)      # 19: 深蓝
    ]
    
    # 应用类别过滤
    filtered_detections = output_data["detections"]
    if filtered_categories:
        # 确保 filtered_categories 是列表
        if isinstance(filtered_categories, str):
            filtered_categories = [filtered_categories]
        filtered_detections = [det for det in output_data["detections"] if det['class'] in filtered_categories]
    
    for det in filtered_detections:
        xmin, ymin = det['bbox']['xmin'], det['bbox']['ymin']
        xmax, ymax = det['bbox']['xmax'], det['bbox']['ymax']
        
        # 根据类别ID获取颜色，如果超出范围则循环使用
        cls_id = det.get('cls_id', 0)
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        
        # 绘制边界框
        cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), color, 4)
        
        # 获取中文标签
        display_name = CLASS_NAME_MAP.get(det['class'], det['class'])  # 直接使用已经转换的中文
        label = f"{display_name} {det['confidence']:.2f}"
        
        # 使用支持中文的绘制方法
        img_with_boxes = draw_chinese_text(
            img_with_boxes,
            label,
            (xmin, ymin-30),
            font_size=30,  # 可根据需要调整大小
            bg_color=color # 传入背景颜色
        )

    # 保存结果图
    cv2.imwrite(output_image_path, img_with_boxes)
    
    return output_image_path, output_json_path

def draw_chinese_text(img, text, pos, font_path='simhei.ttf', font_size=20, bg_color=(0, 255, 0)):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        print(f"⚠️ 字体文件 {font_path} 加载失败，使用默认字体（可能不支持中文）")
    
    # 转换BGR颜色到RGB
    rgb_color = (bg_color[2], bg_color[1], bg_color[0])
    
    # 绘制文字背景（增强可读性）
    text_bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle(text_bbox, fill=rgb_color)
    
    # 绘制文字 (白色文字)
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='YOLOv8目标检测工具')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--model', type=str, default='yolo12n', help='模型路径(.pt文件)或模型名称(如yolo12n)')
    parser.add_argument('--output-img', type=str, help='输出图片路径（可选）')
    parser.add_argument('--output-json', type=str, help='输出JSON路径（可选）')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值（默认0.5）')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU阈值（默认0.5）')
    args = parser.parse_args()

    # 调用检测函数
    try:
        result_img, result_json = yolo_detection(
            image_path=args.image,
            model_path=args.model,
            output_image_path=args.output_img,
            output_json_path=args.output_json,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        print(f"检测完成！\n图片结果: {result_img}\nJSON结果: {result_json}")
    except Exception as e:
        print(f"❌ 检测失败: {str(e)}")