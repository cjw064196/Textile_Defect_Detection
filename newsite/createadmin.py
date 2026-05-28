from django.contrib.auth import get_user_model

# 获取自定义用户模型
CustomUser = get_user_model()

# 创建管理员用户
admin_user = CustomUser.objects.create_user(
    username='admin',  # 用户名
    email='',  # 邮箱
    password='admin',  # 密码
    role='admin'  # 角色为 admin
)

# 设置为超级用户（可选）
admin_user.is_superuser = True
admin_user.is_staff = True
admin_user.save()

print('管理员用户创建成功！')