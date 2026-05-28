from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth import authenticate, login
from .models import CustomUser
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib.auth.hashers import make_password
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import FileSystemStorage
import os
import uuid
from django.conf import settings
import json
from pathlib import Path
from .models import FlowerDetectionRecord,KnowledgeBase
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
import numpy as np
import sys
import os
from django.utils import timezone

# 尝试导入 config_text，确保路径正确
try:
    import config_text
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config_text

# Import after path fix
from yolo_detection import yolo_detection, load_model_with_cache, draw_chinese_text, get_device

# Global dictionary to manage video recording sessions
# Key: session_id, Value: { 'writer': cv2.VideoWriter, 'path': str, 'width': int, 'height': int }
VIDEO_SESSIONS = {}

def login_view(request):
    context = {
        'login_texts': config_text.LOGIN_LEFT_TEXTS,
        'theme_config': config_text.THEME_CONFIG
    }
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            print("user.role:", user.role)
            # 根据角色跳转
            if user.role == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('user_main')
        else:
            context['error'] = '用户名或密码错误'
            return render(request, 'login.html', context)
    return render(request, 'login.html', context)

@login_required
def admin_dashboard(request):
    if not request.user.role == 'admin':
        return redirect('user_dashboard')
    # return render(request, 'admin_dashboard.html')
    return redirect('user_management')

@login_required
def user_main(request):
    context = {
        'main_texts': config_text.MAIN_TEXTS,
        'login_texts': config_text.LOGIN_LEFT_TEXTS,
        'theme_config': config_text.THEME_CONFIG
    }
    return render(request, 'user_main.html', context)



@login_required
def user_management(request):
    if not request.user.role == 'admin':
        return redirect('user_dashboard')
    users = CustomUser.objects.all()
    return render(request, 'user_management.html', {
        'users': users,
        'theme_config': config_text.THEME_CONFIG,
        'login_texts': config_text.LOGIN_LEFT_TEXTS
    })

@login_required
def detection_records(request):
    if not request.user.role == 'admin':
        return redirect('user_dashboard')
    
    records = FlowerDetectionRecord.objects.select_related('user').all().order_by('-detection_time')

    records_data = []
    for record in records:
        result_path = record.image_result_path.replace('\\', '/')
        is_video = result_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))
        
        records_data.append({
            'id': record.id,
            'user': record.user.username,  # 用户名
            'avatar_url': record.user.avatar.url if record.user.avatar else '', # 用户头像
            'image_url': os.path.join(settings.MEDIA_URL, record.image_path).replace('\\', '/'),  # 图片 URL
            'image_result_path': os.path.join(settings.MEDIA_URL, result_path),
            'is_video': is_video,
            'detection_time': record.detection_time.strftime('%Y-%m-%d %H:%M:%S')  # 检测时间
        })

    # print("record data=",records_data)
    return render(request, 'detection_records.html', {
        'records': records_data, 
        'theme_config': config_text.THEME_CONFIG,
        'login_texts': config_text.LOGIN_LEFT_TEXTS
    })

@login_required
@require_POST
def delete_detection_records(request):
    if not request.user.role == 'admin':
        return JsonResponse({'status': 'error', 'message': '权限不足'}, status=403)
        
    try:
        data = json.loads(request.body)
        record_ids = data.get('ids', [])
        
        if not record_ids:
            return JsonResponse({'status': 'error', 'message': '未选择记录'}, status=400)
            
        # Delete records
        # Note: Depending on requirements, we might want to delete associated files too.
        # For now, just deleting DB records as is standard for many simple apps, 
        # but better to delete files if they are not shared.
        # Given the scope, let's just delete the records.
        count, _ = FlowerDetectionRecord.objects.filter(id__in=record_ids).delete()
        
        return JsonResponse({'status': 'success', 'message': f'成功删除 {count} 条记录'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def user_knowledge(request):
    # 获取所有知识库条目并按发布时间降序排列
    knowledge_items = KnowledgeBase.objects.all().order_by('-publish_time')
    
    # 准备前端需要的数据
    knowledge_data = [
        {
            'id': item.id,
            'title': item.title,
            'content': item.content,
            'publish_time': item.publish_time.strftime('%Y-%m-%d %H:%M:%S'),
            'short_content': item.content[:100] + '...' if len(item.content) > 100 else item.content
        }
        for item in knowledge_items
    ]
    
    return render(request, 'user_knowledge.html', {
        'knowledge_list': knowledge_data,
        'theme_config': config_text.THEME_CONFIG,
        'login_texts': config_text.LOGIN_LEFT_TEXTS
    })

@login_required
def knowledge_list(request):
    # 权限检查 - 只有管理员可以访问
    if not request.user.role == 'admin':
        return redirect('user_dashboard')
    
    # 获取所有知识库条目并按发布时间降序排列
    knowledge_items = KnowledgeBase.objects.all().order_by('-publish_time')
    
    # 准备前端需要的数据
    knowledge_data = [
        {
            'id': item.id,
            'title': item.title,
            'content': item.content,
            'publish_time': item.publish_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        for item in knowledge_items
    ]
    
    return render(request, 'knowledge_list.html', {
        'knowledge_list': knowledge_data,  # 修改上下文变量名以匹配模板
        'theme_config': config_text.THEME_CONFIG,
        'login_texts': config_text.LOGIN_LEFT_TEXTS
    })
def logout_view(request):
    from django.contrib.auth import logout
    logout(request)
    return redirect('login')

@csrf_exempt 
def add_user(request):
    print('添加用户')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        avatar = request.FILES.get('avatar')

        # 验证数据
        if not all([username, password, confirm_password]):
            return JsonResponse({'status': 'error', 'message': '所有字段必须填写'}, status=400)
        
        if password != confirm_password:
            return JsonResponse({'status': 'error', 'message': '两次密码输入不一致'}, status=400)
        
        if CustomUser.objects.filter(username=username).exists():
            return JsonResponse({'status': 'error', 'message': '用户名已存在'}, status=400)

        # 创建用户
        user_data = {
            'username': username,
            'password': make_password(password),
            'role': 'user'  # 固定为普通用户
        }
        if avatar:
            user_data['avatar'] = avatar

        user = CustomUser.objects.create(**user_data)
        
        return JsonResponse({'status': 'success', 'message': '用户创建成功'})
    
    return JsonResponse({'status': 'error', 'message': '非法请求'}, status=403)

@csrf_exempt
@login_required
def delete_user(request, user_id):
    if request.method == 'POST' and request.user.role == 'admin':
        try:
            user = CustomUser.objects.get(id=user_id)
            # 禁止删除自己
            if user == request.user:
                return JsonResponse({'status': 'error', 'message': '不能删除当前登录用户'}, status=400)
            user.delete()
            return JsonResponse({'status': 'success'})
        except CustomUser.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': '用户不存在'}, status=404)
    return JsonResponse({'status': 'error', 'message': '非法请求'}, status=403)

@csrf_exempt
@login_required
def update_profile(request):
    if request.method == 'POST':
        try:
            username = request.POST.get('username')
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm_password')
            
            user = request.user
            
            # 修改用户名
            if username and username != user.username:
                if CustomUser.objects.filter(username=username).exclude(id=user.id).exists():
                    return JsonResponse({'status': 'error', 'message': '用户名已存在'}, status=400)
                user.username = username
            
            # 修改密码
            if password:
                if password != confirm_password:
                    return JsonResponse({'status': 'error', 'message': '两次密码输入不一致'}, status=400)
                user.password = make_password(password)

            # 修改头像
            if 'avatar' in request.FILES:
                user.avatar = request.FILES['avatar']
            
            user.save()
            
            # 重新登录以保持会话（如果是修改了密码，Session可能会失效，但这里是手动更新，通常需要重新登录或更新Session hash）
            # 为了简单起见，如果修改了密码，建议前端提示重新登录，或者这里自动更新Session
            login(request, user)
            
            return JsonResponse({
                'status': 'success', 
                'message': '个人信息更新成功',
                'avatar_url': user.avatar.url if user.avatar else ''
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            
    return JsonResponse({'status': 'error', 'message': '非法请求'}, status=405)

@csrf_exempt
@login_required
@require_POST
def edit_user(request, user_id):
    if request.user.role != 'admin':
        return JsonResponse({'status': 'error', 'message': '权限不足'}, status=403)

    try:
        target_user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '用户不存在'}, status=404)

    new_password = request.POST.get('new_password')
    confirm_password = request.POST.get('confirm_password')
    avatar = request.FILES.get('avatar')

    updated = False

    # 修改密码
    if new_password:
        if new_password != confirm_password:
            return JsonResponse({'status': 'error', 'message': '两次密码不一致'}, status=400)
        target_user.set_password(new_password)
        updated = True

    # 修改头像
    if avatar:
        target_user.avatar = avatar
        updated = True

    if updated:
        target_user.save()
        return JsonResponse({'status': 'success', 'message': '用户信息已更新'})
    else:
        return JsonResponse({'status': 'info', 'message': '没有检测到修改内容'})


# @login_required
# @csrf_exempt  
# def temp_image_upload(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp'))
        
#         # 保存文件到临时目录
#         uploaded_file = request.FILES['image']
#         filename = fs.save(uploaded_file.name, uploaded_file)
        
#         return JsonResponse({
#             'status': 'success',
#             'image_url': f"{settings.MEDIA_URL}temp/{filename}"
#         })
    
#     return JsonResponse({'status': 'error'}, status=400)


@require_POST
@csrf_exempt
@login_required
def image_upload(request):
    try:
        # 验证文件存在性
        if 'image' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': '未接收到文件'}, status=400)
        
        uploaded_file = request.FILES['image']
        
        # 文件类型验证
        allowed_types = ['image/jpeg', 'image/png', 'image/gif']
        if uploaded_file.content_type not in allowed_types:
            return JsonResponse({'status': 'error', 'message': '仅支持JPEG/PNG/GIF格式'}, status=415)
        
        # 文件大小验证（5MB）
        max_size = 5 * 1024 * 1024
        if uploaded_file.size > max_size:
            return JsonResponse({'status': 'error', 'message': '文件超过5MB限制'}, status=413)
        print('zou zhege le ')
        # 创建临时存储目录
        temp_storage = FileSystemStorage(
            location=os.path.join(settings.MEDIA_ROOT, 'temp'),
            base_url=os.path.join(settings.MEDIA_URL, 'temp').replace('\\', '/') + '/'
        )
        
        # 生成唯一文件名
        file_ext = os.path.splitext(uploaded_file.name)[1]
        unique_name = f"{uuid.uuid4().hex}{file_ext}"
        
        # 保存文件
        filename = temp_storage.save(unique_name, uploaded_file)
        
        return JsonResponse({
            'status': 'success',
            'image_url': temp_storage.url(filename)
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_POST
@csrf_exempt
@login_required
def upload_video(request):
    try:
        if 'video' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': '未接收到文件'}, status=400)
        
        uploaded_file = request.FILES['video']
        
        # 验证文件类型
        allowed_types = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo']
        if uploaded_file.content_type not in allowed_types and not uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
             # 某些浏览器可能无法正确识别所有视频MIME类型，退回到扩展名检查
             pass
        
        # 文件大小验证（100MB?）
        max_size = 100 * 1024 * 1024
        if uploaded_file.size > max_size:
            return JsonResponse({'status': 'error', 'message': '文件超过100MB限制'}, status=413)
            
        # 创建临时存储目录
        temp_storage = FileSystemStorage(
            location=os.path.join(settings.MEDIA_ROOT, 'temp_videos'),
            base_url=os.path.join(settings.MEDIA_URL, 'temp_videos').replace('\\', '/') + '/'
        )
        
        # 生成唯一文件名
        file_ext = os.path.splitext(uploaded_file.name)[1]
        unique_name = f"{uuid.uuid4().hex}{file_ext}"
        
        filename = temp_storage.save(unique_name, uploaded_file)
        
        return JsonResponse({
            'status': 'success',
            'video_url': temp_storage.url(filename),
            'video_path': os.path.join('temp_videos', filename).replace('\\', '/')
        })
        
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@require_POST
@csrf_exempt
@login_required
def upload_model(request):
    try:
        # Check if file exists
        if 'model_file' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': '未接收到模型文件'}, status=400)
        
        uploaded_file = request.FILES['model_file']
        
        # Check file extension
        if not (uploaded_file.name.endswith('.pt') or uploaded_file.name.endswith('.pth')):
            return JsonResponse({'status': 'error', 'message': '仅支持 .pt 或 .pth 模型文件'}, status=415)
        
        # Save file to models directory
        models_dir = os.path.join(settings.BASE_DIR, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        fs = FileSystemStorage(location=models_dir)
        # Use a safe filename or keep original
        filename = fs.save(uploaded_file.name, uploaded_file)
        
        # Get absolute path
        model_abs_path = os.path.join(models_dir, filename)
        
        # Store in session
        request.session['custom_model_path'] = model_abs_path
        
        return JsonResponse({
            'status': 'success',
            'message': '模型上传成功',
            'filename': filename
        })
        
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
@require_POST
def detect_image(request):
    try:
        print('detact image')
        # 获取前端传递的图片路径
        data = json.loads(request.body)
        image_url = data.get('image_url', '')
        
        is_video_frame = False
        input_abs_path = ""
        input_rel_path = ""

        if image_url.startswith('data:image'):
            # 处理 Base64 图片 (视频/摄像头帧)
            is_video_frame = True
            import base64
            
            # 解析 Base64 数据
            format, imgstr = image_url.split(';base64,') 
            ext = format.split('/')[-1] 
            
            # 创建临时文件
            temp_filename = f"{uuid.uuid4().hex}.{ext}"
            temp_rel_path = os.path.join('temp', temp_filename)
            input_abs_path = os.path.join(settings.MEDIA_ROOT, temp_rel_path)
            
            # 确保存储目录存在
            os.makedirs(os.path.dirname(input_abs_path), exist_ok=True)
            
            # 保存图片
            with open(input_abs_path, 'wb') as f:
                f.write(base64.b64decode(imgstr))
                
            input_rel_path = temp_rel_path # 用于记录
        else:
            # 处理现有逻辑 (服务器上的文件路径)
            if settings.MEDIA_URL in image_url:
                input_rel_path = image_url.split(settings.MEDIA_URL)[1]
            else:
                input_rel_path = image_url

            # Decode URL encoding
            import urllib.parse
            input_rel_path = urllib.parse.unquote(input_rel_path)
            input_abs_path = os.path.join(settings.MEDIA_ROOT, input_rel_path)

        # 获取置信度和IOU阈值
        conf_threshold = float(data.get('conf', 0.5))
        iou_threshold = float(data.get('iou', 0.5))
        filtered_categories = data.get('filtered_categories', None)
        # Get video timestamp if provided
        video_timestamp = data.get('timestamp', None)

        # 配置输出路径
        output_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
        os.makedirs(output_dir, exist_ok=True)

        # Determine model path: Session > Default
        default_model_path = os.path.join(settings.BASE_DIR, 'runs','train_yolo12n','weights','best.pt')
        model_path = request.session.get('custom_model_path', default_model_path)
        
        # Verify model exists, fallback if not
        if not os.path.exists(model_path):
            print(f"Warning: Custom model not found at {model_path}, using default.")
            model_path = default_model_path
        
        # 结果文件路径
        result_filename = os.path.basename(input_rel_path)
        if is_video_frame:
            # 如果是视频帧，为了避免文件名冲突或混乱，也可以用uuid，但input_rel_path已经是uuid了
            pass
            
        try:
            result_img_path, result_json = yolo_detection(
                image_path=input_abs_path,
                model_path=model_path,
                output_image_path=os.path.join(output_dir, result_filename),
                output_json_path=os.path.join(output_dir, f"{Path(input_rel_path).stem}.json"),
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                filtered_categories=filtered_categories
            )
        except Exception as e:
            print(f"YOLO Detection failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

        result_img_rel_path = os.path.relpath(result_img_path, settings.MEDIA_ROOT).replace('\\', '/')
        
        # Handle Video Recording
        recording_session_id = data.get('recording_session_id')
        if recording_session_id and recording_session_id in VIDEO_SESSIONS:
            try:
                session = VIDEO_SESSIONS[recording_session_id]
                # Check if we are using the new buffering mode (has 'temp_dir') or old mode
                if 'temp_dir' in session:
                    import shutil
                    import time
                    # Save frame to temp dir
                    frame_name = f"{time.time()}_{uuid.uuid4().hex[:6]}.jpg"
                    save_path = os.path.join(session['temp_dir'], frame_name)
                    shutil.copy2(result_img_path, save_path)
                else:
                    # Fallback to old writer mode (should not happen after full update)
                    frame = cv2.imread(result_img_path)
                    if frame is not None:
                        writer = session['writer']
                        # Resize to match writer config to prevent errors
                        frame = cv2.resize(frame, (session['width'], session['height']))
                        writer.write(frame)
            except Exception as e:
                print(f"Error writing to video session: {e}")

        with open(result_json, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        # print('result_data:', result_data)

        if is_video_frame:
            # 视频/摄像头模式：不保存记录，不保留文件
            import base64
            
            # 读取结果图片并转换为Base64
            with open(result_img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                processed_image_url = f"data:image/jpeg;base64,{encoded_string}"
            
            # 删除临时文件 (输入图、输出图、JSON结果)
            try:
                if os.path.exists(input_abs_path):
                    os.remove(input_abs_path)
                if os.path.exists(result_img_path):
                    os.remove(result_img_path)
                if os.path.exists(result_json):
                    os.remove(result_json)
            except Exception as e:
                print(f"Error deleting temp files: {e}")

            # 返回结果，不包含history更新
            return JsonResponse({
                'status': 'success',
                'processed_image': processed_image_url,
                # 'history': [], # 不返回历史记录更新
                'detections': result_data['detections'],
                'category_counts': result_data.get('category_counts', {}),
                'inference_time': result_data.get('inference_time', '0ms')
            })

        else:
            # 图片模式：保存记录
            # 生成检测记录
            if request.user.is_authenticated:
                record = FlowerDetectionRecord(
                    user=request.user,  # 当前用户
                    image_path=input_rel_path.replace('\\', '/'),  # 图片路径（相对路径）
                    image_result_path=result_img_rel_path,  # 识别结果
                    video_timestamp=video_timestamp # Save timestamp if available
                )
                record.save()  # 保存记录
                print(f"[DEBUG_SAVE] Saved record ID: {record.id}, Time: {record.detection_time}")

                # 查询当前用户的所有检测记录
                records = FlowerDetectionRecord.objects.filter(user=request.user).order_by('-detection_time')[:20]
                history_data = [
                    {
                        'id': record.id,
                        'image_url': os.path.join(settings.MEDIA_URL, record.image_path).replace('\\', '/'),
                        'image_result_path': os.path.join(settings.MEDIA_URL, record.image_result_path).replace('\\', '/'),
                        'detection_time': timezone.localtime(record.detection_time).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    for record in records
                ]
            else:
                history_data = []

            import time
            processed_image_url = os.path.join(settings.MEDIA_URL, result_img_rel_path).replace('\\', '/') + f"?t={int(time.time())}"
            # print('result_img_path:', processed_image_url)
            return JsonResponse({
                'status': 'success',
                'processed_image': processed_image_url,
                'history': history_data,
                'detections': result_data['detections'],
                'category_counts': result_data.get('category_counts', {}),
                'inference_time': result_data.get('inference_time', '0ms')
            })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*50}")
        print(f"SERVER ERROR in detect_image:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Stack Trace:")
        print(error_trace)
        print(f"{'='*50}\n")
        return JsonResponse({'status': 'error', 'message': str(e), 'error_detail': error_trace}, status=500)




# 检测记录列表
@login_required
@csrf_exempt
def detection_list(request):
    records = FlowerDetectionRecord.objects.select_related('user').all().order_by('-detection_time')
    
    records_data = []
    for record in records:
        result_path = record.image_result_path.replace('\\', '/') if record.image_result_path else ''
        is_video = result_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))
        
        # Determine time to display: video_timestamp (if available) or detection_time
        time_display = timezone.localtime(record.detection_time).strftime('%Y-%m-%d %H:%M:%S')
        if record.video_timestamp is not None:
             import datetime
             # Format seconds to HH:MM:SS
             time_display = str(datetime.timedelta(seconds=int(record.video_timestamp)))

        records_data.append({
            'id': record.id,
            'user': record.user.username,
            'avatar_url': record.user.avatar.url if record.user.avatar else '',
            'image_url': os.path.join(settings.MEDIA_URL, record.image_path).replace('\\', '/'),
            'image_result_path': os.path.join(settings.MEDIA_URL, result_path),
            'is_video': is_video,
            'detection_time': time_display
        })

    # 将封装后的数据传递给模板
    return render(request, 'detection_records.html', {
        'records': records_data,
        'theme_config': config_text.THEME_CONFIG
    })

    # return render(request, 'detection_records.html', {'records': records})





def get_history(request):
    if request.method == 'GET':
        try:
            # 获取当前用户的历史记录，限制为最近20条
            records = FlowerDetectionRecord.objects.filter(user=request.user).order_by('-detection_time')[:20]
            history_data = [
                {
                    'image_url': os.path.join(settings.MEDIA_URL, record.image_path).replace('\\', '/'),
                    'name': record.result,
                    'confidence': record.confidence,
                    'date': timezone.localtime(record.detection_time).strftime('%Y-%m-%d %H:%M:%S')
                }
                for record in records
            ]
            return JsonResponse({'status': 'success', 'history': history_data})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def get_history_html(request):
    if request.method == 'GET':
        try:
            # 获取当前用户的历史记录，限制为最近20条
            records = FlowerDetectionRecord.objects.filter(user=request.user).order_by('-detection_time')[:20]
            # 将 QuerySet 转换为字典列表
            history_data = [
                {
                    'id': record.id,
                    'image_url': os.path.join(settings.MEDIA_URL, record.image_path).replace('\\', '/'),
                    'image_result_path': os.path.join(settings.MEDIA_URL, record.image_result_path).replace('\\', '/'),
                    'detection_time': timezone.localtime(record.detection_time).strftime('%Y-%m-%d %H:%M:%S')
                }
                for record in records
            ]
            
            # print('history_data:', history_data)  # 打印调试信息
            return render(request, 'history_list.html', {'history': history_data})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def search_history(request):
    if request.method == 'GET':
        try:
            # 获取搜索关键词
            keyword = request.GET.get('keyword', '').strip()
            
            # 过滤历史记录
            records = FlowerDetectionRecord.objects.filter(
                user=request.user,
                result__icontains=keyword  # 模糊匹配名
            ).order_by('-detection_time')
            
            # 将 QuerySet 转换为字典列表
            history_data = [
                {
                    'id': record.id,
                    'image_url': os.path.join(settings.MEDIA_URL, record.image_path).replace('\\', '/'),
                    'image_result_path': os.path.join(settings.MEDIA_URL, record.image_result_path).replace('\\', '/'),
                    'detection_time': timezone.localtime(record.detection_time).strftime('%Y-%m-%d %H:%M:%S')
                }
                for record in records
            ]
            print('history_data:', history_data) 
            return render(request, 'history_list.html', {'history': history_data})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # 获取上传的图片
        uploaded_image = request.FILES['image']

        # 保存图片到本地文件系统
        file_path = os.path.join('uploads', uploaded_image.name)
        saved_path = default_storage.save(file_path, ContentFile(uploaded_image.read()))

        # 生成图片的完整 URL
        image_url = request.build_absolute_uri(default_storage.url(saved_path))

        # 返回图片地址
        return JsonResponse({'status': 'success', 'imageUrl': image_url})
    else:
        return JsonResponse({'status': 'error', 'message': '未接收到图片'}, status=400)

@csrf_exempt
@login_required
@require_POST
def process_video_file(request):
    try:
        data = json.loads(request.body)
        video_path = data.get('video_path')
        conf = float(data.get('conf', 0.5))
        iou = float(data.get('iou', 0.5))
        max_duration = float(data.get('max_duration', -1))
        save_history = data.get('save_history', False)
        fast_mode = bool(data.get('fast_mode', False))
        frames = data.get('frames')  # List of frame relative paths under MEDIA_ROOT
        frames_b64 = data.get('frames_b64')  # List of data URLs (base64)
        fps_override = data.get('fps')
        width_override = data.get('width')
        height_override = data.get('height')
        session_id = data.get('session_id')
        duration_override = data.get('duration')
        
        # Debug logging (Temporarily restored for troubleshooting)
        try:
            with open('debug_log.txt', 'a') as f:
                f.write(f"--- Process Video File Request ---\n")
                f.write(f"fast_mode: {fast_mode}, session_id: {session_id}\n")
                f.write(f"duration_override (raw): {duration_override}\n")
                if video_path:
                    f.write(f"video_path: {video_path}\n")
        except Exception as e:
            print(f"Debug log error: {e}")

        # Fast stitching path: stitch existing processed frames without re-detection
        if fast_mode and (frames or frames_b64 or session_id):
            import base64
            import time
            # Determine output filename
            if session_id:
                base_name = f"session_{session_id}"
            elif video_path:
                name, _ = os.path.splitext(os.path.basename(video_path))
                base_name = name
            else:
                base_name = uuid.uuid4().hex
            out_filename = f"{base_name}_stitched.mp4"
            out_rel_path = os.path.join('processed', out_filename)
            out_abs_path = os.path.join(settings.MEDIA_ROOT, out_rel_path)
            os.makedirs(os.path.dirname(out_abs_path), exist_ok=True)
            
            # Build frame list
            images = []
            if frames and isinstance(frames, list):
                for rel in frames:
                    abs_p = os.path.join(settings.MEDIA_ROOT, rel)
                    if os.path.exists(abs_p):
                        img = cv2.imread(abs_p)
                        if img is not None:
                            images.append(img)
            if frames_b64 and isinstance(frames_b64, list):
                for data_url in frames_b64:
                    try:
                        if ',' in data_url:
                            b64 = data_url.split(',', 1)[1]
                        else:
                            b64 = data_url
                        raw = base64.b64decode(b64)
                        np_arr = np.frombuffer(raw, dtype=np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            images.append(img)
                    except Exception as e:
                        print(f"Error decoding frame: {e}")
            
            if session_id and len(images) == 0:
                candidate_rel = os.path.join('processed', f"video_{session_id}.mp4")
                candidate_abs = os.path.join(settings.MEDIA_ROOT, candidate_rel)
                if os.path.exists(candidate_abs):
                    import math
                    src_fps = None
                    src_frame_count = None
                    if video_path:
                        abs_video_path = os.path.join(settings.MEDIA_ROOT, video_path)
                        cap_src = cv2.VideoCapture(abs_video_path)
                        src_fps = cap_src.get(cv2.CAP_PROP_FPS)
                        src_frame_count = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap_src.release()
                    cap_rec = cv2.VideoCapture(candidate_abs)
                    rec_width = int(cap_rec.get(cv2.CAP_PROP_FRAME_WIDTH))
                    rec_height = int(cap_rec.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    rec_frame_count = int(cap_rec.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Log FPS info
                    try:
                        with open('debug_log.txt', 'a') as f:
                            f.write(f"src_fps: {src_fps}, src_frame_count: {src_frame_count}\n")
                            f.write(f"fps_override: {fps_override}, cap_rec_fps: {cap_rec.get(cv2.CAP_PROP_FPS)}\n")
                    except: pass

                    fps_out = float(src_fps) if src_fps and src_fps > 0 else float(fps_override or cap_rec.get(cv2.CAP_PROP_FPS) or 20.0)
                    
                    if duration_override and float(duration_override) > 0:
                        frames_needed = int(max(1, round(float(duration_override) * fps_out)))
                        try:
                            with open('debug_log.txt', 'a') as f:
                                f.write(f"Using duration_override: {duration_override}, fps_out: {fps_out} -> frames_needed: {frames_needed}\n")
                        except: pass
                    elif src_fps and src_fps > 0 and src_frame_count and src_frame_count > 0:
                        frames_needed = int(max(1, round((src_frame_count / src_fps) * fps_out)))
                        try:
                            with open('debug_log.txt', 'a') as f:
                                f.write(f"Using source duration. frames_needed: {frames_needed}\n")
                        except: pass
                    else:
                        frames_needed = rec_frame_count
                        try:
                            with open('debug_log.txt', 'a') as f:
                                f.write(f"Using recording frame count. frames_needed: {frames_needed}\n")
                        except: pass
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    except:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_filename = f"session_{session_id}_stitched.mp4"
                    out_rel_path = os.path.join('processed', out_filename)
                    out_abs_path = os.path.join(settings.MEDIA_ROOT, out_rel_path)
                    os.makedirs(os.path.dirname(out_abs_path), exist_ok=True)
                    writer = cv2.VideoWriter(out_abs_path, fourcc, fps_out, (rec_width, rec_height))
                    if not writer.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(out_abs_path, fourcc, fps_out, (rec_width, rec_height))
                    if not writer.isOpened():
                        cap_rec.release()
                        return JsonResponse({'status': 'error', 'message': 'Failed to open VideoWriter'}, status=500)
                    if rec_frame_count <= 0:
                        cap_rec.release()
                        writer.release()
                        return JsonResponse({'status': 'error', 'message': 'No recorded frames'}, status=400)
                    for j in range(rec_frame_count):
                        ret, frame = cap_rec.read()
                        if not ret:
                            break
                        write_count = int(math.floor((j + 1) * frames_needed / rec_frame_count) - math.floor(j * frames_needed / rec_frame_count))
                        for _ in range(max(0, write_count)):
                            writer.write(frame)
                    cap_rec.release()
                    writer.release()
                    video_url = os.path.join(settings.MEDIA_URL, out_rel_path).replace('\\', '/')
                    if save_history and request.user.is_authenticated:
                        try:
                            thumb_filename = f"thumb_{session_id}.jpg"
                            thumb_rel_path = os.path.join('processed', thumb_filename)
                            thumb_abs_path = os.path.join(settings.MEDIA_ROOT, thumb_rel_path)
                            # Prefer source video first frame for thumbnail
                            frame_for_thumb = None
                            if video_path:
                                abs_video_src = os.path.join(settings.MEDIA_ROOT, video_path)
                                cap_src = cv2.VideoCapture(abs_video_src)
                                ret_src, frame_src = cap_src.read()
                                if ret_src:
                                    frame_for_thumb = frame_src
                                cap_src.release()
                            if frame_for_thumb is None:
                                cap_thumb = cv2.VideoCapture(candidate_abs)
                                ret_thumb, frame_thumb = cap_thumb.read()
                                if ret_thumb:
                                    frame_for_thumb = frame_thumb
                                cap_thumb.release()
                            if frame_for_thumb is not None:
                                cv2.imwrite(thumb_abs_path, frame_for_thumb)
                                record = FlowerDetectionRecord(
                                    user=request.user,
                                    image_path=thumb_rel_path.replace('\\', '/'),
                                    image_result_path=out_rel_path.replace('\\', '/'),
                                )
                                record.save()
                        except Exception as e:
                            print(f"Error saving history in fast_mode shortcut: {e}")
                    
                    # Clean up the temporary recording file to avoid storage waste
                    try:
                        if os.path.exists(candidate_abs):
                            os.remove(candidate_abs)
                    except Exception as e:
                        print(f"Error cleaning up temp file in fast_mode: {e}")

                    return JsonResponse({'status': 'success', 'video_url': video_url, 'fps': fps_out})
            
            if len(images) == 0:
                return JsonResponse({'status': 'error', 'message': 'No frames available for stitching'}, status=400)
            
            import math
            h0, w0 = images[0].shape[:2]
            width = int(width_override or w0)
            height = int(height_override or h0)
            fps = float(fps_override or 20.0)
            src_fps = None
            src_frame_count = None
            frames_needed = None
            if video_path:
                abs_video_path = os.path.join(settings.MEDIA_ROOT, video_path)
                cap_src = cv2.VideoCapture(abs_video_path)
                src_fps = cap_src.get(cv2.CAP_PROP_FPS)
                src_frame_count = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_src.release()
                if src_fps and src_fps > 0 and src_frame_count and src_frame_count > 0:
                    fps = float(src_fps)
                    if duration_override and float(duration_override) > 0:
                        frames_needed = int(max(1, round(float(duration_override) * fps)))
                    else:
                        frames_needed = int(max(1, round((src_frame_count / src_fps) * fps)))
            
            # Init writer
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_abs_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_abs_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                return JsonResponse({'status': 'error', 'message': 'Failed to open VideoWriter'}, status=500)
            
            if frames_needed and frames_needed > 0:
                m = len(images)
                for j in range(m):
                    img = images[j]
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height))
                    write_count = int(math.floor((j + 1) * frames_needed / m) - math.floor(j * frames_needed / m))
                    for _ in range(max(0, write_count)):
                        writer.write(img)
            else:
                for img in images:
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height))
                    writer.write(img)
            writer.release()
            
            video_url = os.path.join(settings.MEDIA_URL, out_rel_path).replace('\\', '/')
            
            # Save history if requested
            if save_history and request.user.is_authenticated:
                try:
                    thumb_filename = f"thumb_{base_name}.jpg"
                    thumb_rel_path = os.path.join('processed', thumb_filename)
                    thumb_abs_path = os.path.join(settings.MEDIA_ROOT, thumb_rel_path)
                    # Prefer thumbnail from SOURCE video first frame if available
                    frame_for_thumb = None
                    if video_path:
                        abs_video_src = os.path.join(settings.MEDIA_ROOT, video_path)
                        cap_src = cv2.VideoCapture(abs_video_src)
                        ret_src, frame_src = cap_src.read()
                        if ret_src:
                            frame_for_thumb = frame_src
                        cap_src.release()
                    # Fallback to first processed image
                    if frame_for_thumb is None and len(images) > 0:
                        frame_for_thumb = images[0]
                    if frame_for_thumb is not None:
                        cv2.imwrite(thumb_abs_path, frame_for_thumb)
                        record = FlowerDetectionRecord(
                            user=request.user,
                            image_path=thumb_rel_path.replace('\\', '/'),
                            image_result_path=out_rel_path.replace('\\', '/'),
                        )
                        record.save()
                except Exception as e:
                    print(f"Error saving history in fast_mode: {e}")
            
            return JsonResponse({'status': 'success', 'video_url': video_url, 'fps': fps})
        
        # Original high-accuracy path: re-run detection over source video
        if not video_path:
            return JsonResponse({'status': 'error', 'message': 'Missing video_path'}, status=400)
            
        abs_video_path = os.path.join(settings.MEDIA_ROOT, video_path)
        if not os.path.exists(abs_video_path):
             return JsonResponse({'status': 'error', 'message': 'File not found'}, status=404)
             
        cap = cv2.VideoCapture(abs_video_path)
        if not cap.isOpened():
             return JsonResponse({'status': 'error', 'message': 'Cannot open video'}, status=500)
             
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        out_filename = f"{name}_processed.mp4"
        out_rel_path = os.path.join('processed', out_filename)
        out_abs_path = os.path.join(settings.MEDIA_ROOT, out_rel_path)
        os.makedirs(os.path.dirname(out_abs_path), exist_ok=True)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_abs_path, fourcc, fps, (width, height))
        if not writer.isOpened():
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             writer = cv2.VideoWriter(out_abs_path, fourcc, fps, (width, height))
        
        default_model_path = os.path.join(settings.BASE_DIR, 'runs','train_yolo12n','weights','best.pt')
        model_path = request.session.get('custom_model_path', default_model_path)
        if not os.path.exists(model_path):
            model_path = default_model_path
        model = load_model_with_cache(model_path, conf)
        
        from yolo_detection import get_device
        device = get_device()
        print(f"Video processing using device: {device}")
        
        classes_indices = None
        filtered_categories = data.get('filtered_categories')
        if filtered_categories:
            try:
                model_names = model.names
                classes_indices = [k for k, v in model_names.items() if v in filtered_categories]
            except:
                pass
 
        COLOR_PALETTE = [
            (51, 87, 255),
            (87, 255, 51),
            (255, 87, 51),
            (255, 51, 243),
            (168, 51, 255),
            (246, 255, 51),
            (51, 214, 255),
            (255, 51, 168),
            (51, 51, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 0, 255),
            (0, 128, 255),
            (128, 0, 128),
            (128, 128, 0),
            (0, 128, 128),
            (128, 0, 0)
        ]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_duration > 0:
                current_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if current_pos_sec > max_duration:
                    break
            results = model.predict(source=frame, conf=conf, iou=iou, classes=classes_indices, verbose=False, device=device)
            annotated_frame = frame.copy()
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    conf_val = float(box.conf)
                    class_name = result.names[cls_id]
                    chinese_name = config_text.CLASS_NAME_MAP.get(class_name, class_name)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{chinese_name} {conf_val:.2f}"
                    annotated_frame = draw_chinese_text(
                        annotated_frame,
                        label,
                        (x1, y1 - 30),
                        font_size=30,
                        bg_color=color
                    )
            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                annotated_frame = cv2.resize(annotated_frame, (width, height))
            writer.write(annotated_frame)
            
        cap.release()
        writer.release()
        
        video_url = os.path.join(settings.MEDIA_URL, out_rel_path).replace('\\', '/')
        
        # Save detection record if requested and user is authenticated
        if save_history and request.user.is_authenticated:
            print(f"Saving history for user {request.user.username}, video: {name}")
            try:
                # Generate thumbnail from the first frame of the SOURCE video (Original)
                thumb_filename = f"thumb_{name}.jpg"
                thumb_rel_path = os.path.join('processed', thumb_filename)
                thumb_abs_path = os.path.join(settings.MEDIA_ROOT, thumb_rel_path)
                
                # Check if we can capture first frame from SOURCE video
                cap_source = cv2.VideoCapture(abs_video_path)
                ret_src, frame_src = cap_source.read()
                
                if ret_src:
                    cv2.imwrite(thumb_abs_path, frame_src)
                else:
                    print("Warning: Could not read frame from source video for thumbnail. Trying processed video.")
                    # Fallback to processed video
                    cap_processed = cv2.VideoCapture(out_abs_path)
                    ret_thumb, frame_thumb = cap_processed.read()
                    if ret_thumb:
                        cv2.imwrite(thumb_abs_path, frame_thumb)
                    cap_processed.release()
                
                cap_source.release()
                
                # Create Record
                record = FlowerDetectionRecord(
                    user=request.user,
                    image_path=thumb_rel_path.replace('\\', '/'), # Use thumbnail
                    image_result_path=out_rel_path.replace('\\', '/'),
                )
                record.save()
                print(f"History record saved: {record}")
            except Exception as e:
                print(f"Error saving history record in process_video_file: {e}")
                # We don't fail the request just because history save failed
        else:
            print(f"Skipping history save. save_history={save_history}, auth={request.user.is_authenticated}")
        
        return JsonResponse({
            'status': 'success',
            'video_url': video_url,
            'fps': fps
        })
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@require_POST
@csrf_exempt
@login_required
def manage_video_recording(request):
    try:
        data = json.loads(request.body)
        action = data.get('action')
        session_id = data.get('session_id')

        if action == 'start':
            width = int(data.get('width', 640))
            height = int(data.get('height', 480))
            # fps is now just a hint or fallback
            fps = float(data.get('fps', 20.0))
            
            if not session_id:
                session_id = uuid.uuid4().hex
            
            # Create temp directory for frames
            temp_dir_name = f"session_{session_id}"
            temp_dir_rel = os.path.join('temp_frames', temp_dir_name)
            temp_dir_abs = os.path.join(settings.MEDIA_ROOT, temp_dir_rel)
            os.makedirs(temp_dir_abs, exist_ok=True)
            
            # Create video file path (for final output)
            video_filename = f"video_{session_id}.mp4"
            video_rel_path = os.path.join('processed', video_filename)
            video_abs_path = os.path.join(settings.MEDIA_ROOT, video_rel_path)
            os.makedirs(os.path.dirname(video_abs_path), exist_ok=True)
            
            # Store session info
            VIDEO_SESSIONS[session_id] = {
                'temp_dir': temp_dir_abs, # Flag to use temp dir buffering
                'path': video_rel_path,
                'width': width,
                'height': height,
                'fps': fps # Store default fps
            }
            
            return JsonResponse({'status': 'success', 'session_id': session_id})

        if action == 'stop':
            if not session_id:
                 return JsonResponse({'status': 'error', 'message': 'Missing session_id'}, status=400)
            
            skip_save = data.get('skip_save', False)
            delete_temp_file = data.get('delete_temp_file', True)
            duration = data.get('duration') # Get duration from frontend

            if session_id in VIDEO_SESSIONS:
                session = VIDEO_SESSIONS[session_id]
                
                # Check if we are using temp_dir mode
                if 'temp_dir' in session:
                    temp_dir = session['temp_dir']
                    video_rel_path = session['path']
                    video_abs_path = os.path.join(settings.MEDIA_ROOT, video_rel_path)
                    width = session['width']
                    height = session['height']
                    default_fps = session['fps']
                    
                    # 1. Collect all frames
                    import glob
                    files = glob.glob(os.path.join(temp_dir, "*.jpg"))
                    # Sort by filename (timestamp)
                    files.sort()
                    
                    if not files:
                        print("Warning: No frames captured in session")
                    else:
                        # 2. Calculate FPS
                        # If duration is provided and valid, use it to calculate FPS
                        if duration and float(duration) > 0:
                            calculated_fps = len(files) / float(duration)
                            # Clamp FPS to reasonable limits if needed, or just use it
                            final_fps = max(0.1, calculated_fps) # Allow low fps
                            print(f"Session {session_id}: {len(files)} frames, duration {duration}s -> FPS {final_fps}")
                        else:
                            final_fps = default_fps
                            print(f"Session {session_id}: Duration invalid, using default FPS {final_fps}")

                        # 3. Write Video
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                        except:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            
                        writer = cv2.VideoWriter(video_abs_path, fourcc, final_fps, (width, height))
                        if not writer.isOpened():
                             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                             writer = cv2.VideoWriter(video_abs_path, fourcc, final_fps, (width, height))
                        
                        if writer.isOpened():
                            for fpath in files:
                                frame = cv2.imread(fpath)
                                if frame is not None:
                                    # Resize if needed (though should be correct from detect_image)
                                    if frame.shape[1] != width or frame.shape[0] != height:
                                        frame = cv2.resize(frame, (width, height))
                                    writer.write(frame)
                            writer.release()
                        else:
                            print("Error: Could not open VideoWriter for final stitching")

                    # 4. Cleanup Temp Dir
                    import shutil
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        print(f"Error removing temp dir: {e}")

                else:
                    # Old mode (VideoWriter was open) - Should not happen with new start logic
                    # But for safety, keep existing close logic
                    writer = session.get('writer')
                    if writer:
                        writer.release()
                    video_rel_path = session['path']

                video_url = os.path.join(settings.MEDIA_URL, video_rel_path).replace('\\', '/')
                
                # Save detection record if user is authenticated AND not skipping save
                if request.user.is_authenticated and not skip_save:
                    try:
                        video_abs_path = os.path.join(settings.MEDIA_ROOT, video_rel_path)
                        
                        # Generate thumbnail
                        thumbnail_rel_path = video_rel_path # Fallback
                        try:
                            cap = cv2.VideoCapture(video_abs_path)
                            ret, frame = cap.read()
                            if ret:
                                thumb_filename = f"thumb_{session_id}.jpg"
                                thumb_rel_path = os.path.join('processed', thumb_filename)
                                thumb_abs_path = os.path.join(settings.MEDIA_ROOT, thumb_rel_path)
                                cv2.imwrite(thumb_abs_path, frame)
                                thumbnail_rel_path = thumb_rel_path.replace('\\', '/')
                            cap.release()
                        except Exception as e:
                            print(f"Error generating thumbnail: {e}")

                        # Create Record
                        record = FlowerDetectionRecord(
                            user=request.user,
                            image_path=thumbnail_rel_path, # Use thumbnail as 'original' image
                            image_result_path=video_rel_path.replace('\\', '/'),
                        )
                        record.save()
                    except Exception as e:
                        print(f"Error saving video record: {e}")
                        import traceback
                        traceback.print_exc()
                elif skip_save:
                    # If skipping save (e.g. video mode where we use high-quality process instead),
                    # we might want to clean up the temporary recording file to avoid accumulation
                    if delete_temp_file:
                        try:
                            video_abs_path = os.path.join(settings.MEDIA_ROOT, video_rel_path)
                            if os.path.exists(video_abs_path):
                                os.remove(video_abs_path)
                            # Also do not return video_url as it is no longer valid
                            video_url = None
                        except Exception as e:
                            print(f"Error cleaning up skipped video session: {e}")
                    else:
                        # If NOT deleting, we still return the video_url so it can be used
                        pass

                # Clean up session
                del VIDEO_SESSIONS[session_id]
                
                return JsonResponse({'status': 'success', 'video_url': video_url})
            else:
                # Session not found - likely already stopped or never started
                # Instead of 500 or 404 error, we should return success or a specific code
                # to prevent frontend from treating it as a critical failure
                print(f"Warning: Stop requested for non-existent session {session_id}")
                return JsonResponse({'status': 'success', 'message': 'Session already closed or invalid'})
        
        return JsonResponse({'status': 'error', 'message': 'Invalid action'}, status=400)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if not username or not password:
            return render(request, 'register.html', {'error': '用户名和密码不能为空', 'theme_config': config_text.THEME_CONFIG})

        if password != confirm_password:
            return render(request, 'register.html', {'error': '两次密码输入不一致', 'theme_config': config_text.THEME_CONFIG})
        
        if CustomUser.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error': '用户名已存在', 'theme_config': config_text.THEME_CONFIG})
            
        try:
            # create_user will handle password hashing and default avatar
            user = CustomUser.objects.create_user(username=username, password=password)
            # Default role is 'user', default avatar is set in model
            user.save()
            return redirect('login')
        except Exception as e:
            return render(request, 'register.html', {'error': f'注册失败: {str(e)}', 'theme_config': config_text.THEME_CONFIG})

    return render(request, 'register.html', {'theme_config': config_text.THEME_CONFIG})

@csrf_exempt
def get_detection_categories(request):
    """
    获取所有检测类别
    """
    try:
        # 从配置文件导入CLASS_NAME_MAP
        from config_text import CLASS_NAME_MAP
        
        # 提取所有中文类别名称
        categories = list(CLASS_NAME_MAP.values())
        
        return JsonResponse({
            'status': 'success',
            'categories': categories
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@login_required
@csrf_exempt  # 如果使用AJAX需要添加
def add_knowledge(request):
    if request.method == 'POST':
        if not request.user.role == 'admin':
            return JsonResponse({'status': 'error', 'message': '权限不足'}, status=403)
        
        title = request.POST.get('title')
        content = request.POST.get('content')
        
        if not title or not content:
            return JsonResponse({'status': 'error', 'message': '标题和内容不能为空'}, status=400)
        
        try:
            
            new_knowledge = KnowledgeBase.objects.create(
                title=title,
                content=content,
            )
            return JsonResponse({
                'status': 'success',
                'message': '添加成功',
                'data': {
                    'id': new_knowledge.id,
                    'title': new_knowledge.title,
                    'publish_time': new_knowledge.publish_time.strftime('%Y-%m-%d %H:%M:%S')
                }
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'添加失败: {str(e)}'}, status=500)
    
    return JsonResponse({'status': 'error', 'message': '无效请求'}, status=400)

@login_required
@csrf_exempt
def update_knowledge(request):
    if request.method == 'POST':
        if not request.user.role == 'admin':
            return JsonResponse({'status': 'error', 'message': '权限不足'}, status=403)
        
        knowledge_id = request.POST.get('id')
        title = request.POST.get('title')
        content = request.POST.get('content')
        
        if not all([knowledge_id, title, content]):
            return JsonResponse({'status': 'error', 'message': '参数不完整'}, status=400)
        
        try:
            knowledge = KnowledgeBase.objects.get(id=knowledge_id)
            knowledge.title = title
            knowledge.content = content
            knowledge.save()
            
            return JsonResponse({
                'status': 'success',
                'message': '更新成功',
                'data': {
                    'id': knowledge.id,
                    'title': knowledge.title,
                    'content': knowledge.content
                }
            })
        except KnowledgeBase.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': '知识条目不存在'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'更新失败: {str(e)}'}, status=500)
    
    return JsonResponse({'status': 'error', 'message': '无效请求'}, status=400)

@login_required
@csrf_exempt
def delete_knowledge(request):
    if request.method == 'POST':
        if not request.user.role == 'admin':
            return JsonResponse({'status': 'error', 'message': '权限不足'}, status=403)
        
        knowledge_id = request.POST.get('id')
        
        if not knowledge_id:
            return JsonResponse({'status': 'error', 'message': '缺少知识ID'}, status=400)
        
        try:
            knowledge = KnowledgeBase.objects.get(id=knowledge_id)
            knowledge.delete()
            return JsonResponse({'status': 'success', 'message': '删除成功'})
        except KnowledgeBase.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': '知识条目不存在'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'删除失败: {str(e)}'}, status=500)
    
    return JsonResponse({'status': 'error', 'message': '无效请求'}, status=400)
