"""
URL configuration for newsite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.shortcuts import redirect
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', lambda request: redirect('login'), name='root'),
    path('login/', views.login_view, name='login'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),

    path('user/main/', views.user_main, name='user_main'),
    path('logout/', views.logout_view, name='logout'),
    path('admin/users/', views.user_management, name='user_management'),
    path('admin/records/', views.detection_records, name='detection_records'),
    path('admin/knowledge_list/', views.knowledge_list, name='knowledge_list'),
    path('admin/users/add/', views.add_user, name='add_user'),
    path('admin/users/<int:user_id>/delete/', views.delete_user, name='delete_user'),
    path('admin/users/<int:user_id>/edit/', views.edit_user, name='edit_user'),
    path('update_profile/', views.update_profile, name='update_profile'),
    path('upload/', views.image_upload, name='image_upload'),
    path('detect/', views.detect_image, name='detect'),
    path('detection/', views.detection_list, name='detection_list'),
    path('admin/detection/delete/', views.delete_detection_records, name='delete_detection_records'),
    path('get_history/', views.get_history, name='get_history'),
    path('get_history_html/', views.get_history_html, name='get_history_html'),  # 新增HTML片段接口
    path('search_history/', views.search_history, name='search_history'),
    path('upload2/', views.upload_image, name='upload_image'),
    path('register/', views.register_view, name='register'),
    path('add_knowledge/', views.add_knowledge, name='add_knowledge'),
    path('update_knowledge/', views.update_knowledge, name='update_knowledge'),
    path('delete_knowledge/', views.delete_knowledge, name='delete_knowledge'),
    path('user_knowledge/', views.user_knowledge, name='user_knowledge'),
    path('get_categories/', views.get_detection_categories, name='get_categories'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('process_video_file/', views.process_video_file, name='process_video_file'),
    path('upload_model/', views.upload_model, name='upload_model'),
    path('manage_video_recording/', views.manage_video_recording, name='manage_video_recording'), # Video recording management
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
