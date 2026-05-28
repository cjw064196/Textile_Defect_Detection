# newsite/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.contrib.auth import get_user_model

class CustomUser(AbstractUser):
    # 添加角色字段（admin/user）
    ROLE_CHOICES = (
        ('admin', '管理员'),
        ('user', '普通用户'),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')
    avatar = models.ImageField(upload_to='avatars/', default='avatars/default.jpg', verbose_name='头像')


User = get_user_model()

class FlowerDetectionRecord(models.Model):
    """
    检测记录模型
    """
    # 检测用户（外键，关联到 Django 内置的用户模型）
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="检测用户")

    # 检测的图片路径（字符串字段）
    image_path = models.CharField(max_length=255, verbose_name="图片路径")

    # 识别结果（名称）
    image_result_path = models.CharField(max_length=255, verbose_name="结果图片路径")

    # 视频时间戳（秒），用于视频检测
    video_timestamp = models.FloatField(null=True, blank=True, verbose_name="视频时间戳")

    # 检测时间（自动记录检测时间）
    detection_time = models.DateTimeField(auto_now_add=True, verbose_name="检测时间")

    def __str__(self):
        """
        返回模型的字符串表示
        """
        return f"{self.user.username} - {self.result} - {self.detection_time}"

    class Meta:
        verbose_name = "检测记录"
        verbose_name_plural = "检测记录"

class KnowledgeBase(models.Model):
    """
    知识库模型（仅包含标题、内容和发布时间）
    """
    # 知识标题
    title = models.CharField(max_length=200, verbose_name="标题")
    
    # 知识内容
    content = models.TextField(verbose_name="内容")
    
    # 发布时间（自动设置为当前时间）
    publish_time = models.DateTimeField(auto_now_add=True, verbose_name="发布时间")

    def __str__(self):
        return self.title
    
    class Meta:
        verbose_name = "知识库"
        verbose_name_plural = "知识库"

