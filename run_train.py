# -*- coding: utf-8 -*-
"""
该脚本用于执行YOLO模型的训练。

它会自动处理以下任务：
1. 动态修改数据集配置文件 (data.yaml)，将相对路径更新为绝对路径，以确保训练时能正确找到数据。
2. 从 'pretrained' 文件夹加载指定的预训练模型。
3. 使用预设的参数（如epochs, imgsz, batch）启动训练过程。

要开始训练，只需直接运行此脚本。
"""
import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def main():
    """
    主训练函数。
    
    该函数负责执行YOLO模型的训练流程，包括：
    1. 配置预训练模型。
    2. 动态修改数据集的YAML配置文件，确保路径为绝对路径。
    3. 加载预训练模型。
    4. 使用指定参数开始训练。
    """
    # --- 1. 配置模型和路径 ---
    
    # 要训练的模型列表
    models_to_train = [
        {'name': 'yolo12n.pt', 'train_name': 'train_yolo12n'}
    ]
    
    # 获取当前工作目录的绝对路径，以避免相对路径带来的问题
    current_dir = os.path.abspath(os.getcwd())
    
    # --- 2. 动态配置数据集YAML文件 ---
    
    # 构建数据集yaml文件的绝对路径
    data_yaml_path = os.path.join(current_dir, 'train_data', 'data.yaml')
    
    # 读取原始yaml文件内容
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 将yaml文件中的 'path' 字段修改为数据集目录的绝对路径
    # 这是为了确保ultralytics库能正确定位到训练、验证和测试集
    data_config['path'] = os.path.join(current_dir, 'train_data')
    
    # 将修改后的配置写回yaml文件
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    # --- 3. 循环训练每个模型 ---
    
    for model_info in models_to_train:
        model_name = model_info['name']
        train_name = model_info['train_name']
        
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_name}")
        print(f"训练名称: {train_name}")
        print(f"{'='*60}")
        
        # 构建预训练模型的完整路径
        pretrained_model_path = os.path.join(current_dir, 'pretrained', model_name)
        if not os.path.exists(pretrained_model_path):
            print(f"警告: 预训练模型文件不存在: {pretrained_model_path}")
            print(f"跳过模型 {model_name} 的训练")
            continue
        
        try:
            # 加载指定的预训练模型
            model = YOLO(pretrained_model_path)
            
            # --- 4. 开始训练 ---
            
            print(f"开始训练 {model_name}...")
            # 调用train方法开始训练
            model.train(
                data=data_yaml_path,  # 数据集配置文件
                epochs=100,           # 训练轮次
                imgsz=640,            # 输入图像尺寸
                batch=8,             # 每批次的图像数量
                name=train_name,      # 模型名称
            )
            
            print(f"{model_name} 训练完成！")
            
        except Exception as e:
            print(f"训练 {model_name} 时出现错误: {str(e)}")
            print(f"跳过模型 {model_name}，继续训练下一个模型")
            continue
    
    print(f"\n{'='*60}")
    print("所有模型训练完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 当该脚本被直接执行时，调用main函数
    main()
