# -*- coding: utf-8 -*-
"""
该脚本作为应用的文本配置文件。

它集中管理了界面中所有需要显示的静态文本，如窗口标题、标签、描述等。
通过修改此文件中的字典，可以方便地实现界面语言的切换或内容的自定义，
而无需修改主程序的代码。
"""

# 登录界面左侧文本配置
LOGIN_LEFT_TEXTS = {  # 用于登录界面的标题、副标题和描述
    "title": "纺织品缺陷检测系统",
    "subtitle": "纺织品缺陷检测 · 智能识别", 
    "description": "基于深度学习的纺织品缺陷检测系统\n实时识别纺织品缺陷"
}

# 主界面文本配置
MAIN_TEXTS = {  # 用于主窗口的标题
    "title": "基于YOLO算法的纺织品缺陷检测系统",
    "panel_title": "纺织品缺陷检测"
}

# 全局类别名称映射
CLASS_NAME_MAP = {
    'delik': '洞',
    'dokuma_iplik_hata': '织线瑕疵',
    'leke': '污渍',
    'topbasi': '纱头'
}

# 界面风格配置
THEME_CONFIG = {
    "theme_css": "css/themes/tech-dark-1.css",  # 主题样式文件
    "anim_css": "css/animations/tech-anim-1.css" # 动画样式文件
}
