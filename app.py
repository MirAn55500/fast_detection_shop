from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import os
import tempfile
import shutil
from person_detection_service import ObjectDetectionService
import json
import asyncio
import time

app = FastAPI(title="Object Detection Service")

# Create directories for static files and event frames
os.makedirs("static", exist_ok=True)
os.makedirs("static/event_frames", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Global state to track processing progress
processing_status = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def process_video_task(file_location, model_type, task_id):
    """Background task for video processing"""
    try:
        # Initialize status
        processing_status[task_id] = {
            "progress": 0,
            "status": "processing",
            "message": "Starting processing...",
            "file_path": file_location
        }
        
        # Process video with detection service
        use_custom_model = (model_type == "custom")
        service = ObjectDetectionService(use_custom_model=use_custom_model)
        
        # Set up progress callback
        def progress_callback(current_progress, max_progress, message="Processing..."):
            progress = min(int((current_progress / max_progress) * 100), 100)
            processing_status[task_id].update({
                "progress": progress,
                "status": "processing",
                "message": message
            })
        
        # Redirect event frames to the static directory
        service.event_frames_dir = "static/event_frames"
        service.progress_callback = progress_callback
        
        # Process the video
        results_df = service.process_video(file_location)
        
        # Check if detection video was created
        detection_video_path = "static/detection_video.mp4"
        has_detection_video = os.path.exists(detection_video_path)
        if has_detection_video:
            video_size = os.path.getsize(detection_video_path)
            print(f"Detection video created: {detection_video_path}, size: {video_size} bytes")
            
            # Проверка, что видео полностью записано
            if video_size < 1000:  # Если файл слишком маленький, возможно ошибка
                print(f"Warning: Detection video too small: {video_size} bytes")
                has_detection_video = False
        else:
            print(f"Warning: Detection video not found at {detection_video_path}")
        
        # Update status to complete
        results_list = []
        if not results_df.empty:
            # Convert DataFrame to list of dictionaries
            results_list = results_df.to_dict(orient='records')
            
            # Get statistics for display
            stats = {}
            if 'event_type' in results_df.columns:
                stats = results_df['event_type'].value_counts().to_dict()
        
        # Добавляем задержку перед тем, как отметить обработку как завершенную
        # Это дает время для завершения записи видео
        time.sleep(3)
        
        processing_status[task_id].update({
            "progress": 100,
            "status": "complete",
            "message": "Processing complete",
            "results": results_list,
            "stats": stats,
            "file_path": file_location,
            "has_detection_video": has_detection_video,
            "completion_time": time.time()  # Добавляем время завершения
        })
            
    except Exception as e:
        print(f"Error in process_video_task: {e}")
        import traceback
        traceback.print_exc()
        processing_status[task_id].update({
            "progress": 0,
            "status": "error",
            "message": f"Error: {str(e)}"
        })

@app.post("/process", response_class=HTMLResponse)
async def process_video(request: Request, file: UploadFile = File(...), model_type: str = Form(...), background_tasks: BackgroundTasks = None):
    # Generate unique ID for this task
    task_id = f"task_{int(time.time())}"
    
    # Save uploaded file
    file_location = f"static/uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Start processing in background
    background_tasks.add_task(process_video_task, file_location, model_type, task_id)
    
    # Return processing page with task ID
    return templates.TemplateResponse(
        "processing.html", 
        {
            "request": request,
            "task_id": task_id,
            "file_name": file.filename
        }
    )

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Return current progress of a task"""
    if task_id not in processing_status:
        return {"progress": 0, "status": "not_found"}
    return processing_status[task_id]

@app.get("/wait-for-video/{task_id}")
async def wait_for_video(task_id: str):
    """Endpoint, который будет проверять готовность видео перед редиректом"""
    # Проверяем наличие видео
    # Проверяем оба возможных формата видео (mp4 и avi)
    video_paths = ["static/detection_video.mp4", "static/detection_video.avi"]
    detection_video_path = None
    video_size = 0
    
    for path in video_paths:
        if os.path.exists(path):
            detection_video_path = path
            video_size = os.path.getsize(path)
            break
    
    # Проверяем статус задачи
    if task_id not in processing_status or processing_status[task_id]["status"] != "complete":
        return {"ready": False, "message": "Processing not complete"}
    
    # Проверяем, прошло ли достаточно времени с момента завершения
    completion_time = processing_status[task_id].get("completion_time", 0)
    elapsed = time.time() - completion_time
    
    # Формируем ответ
    return {
        "ready": detection_video_path is not None and video_size > 1000 and elapsed > 2,
        "has_video": detection_video_path is not None,
        "video_size": video_size,
        "elapsed": elapsed,
        "video_path": detection_video_path
    }

@app.get("/results/{task_id}", response_class=HTMLResponse)
async def get_results(request: Request, task_id: str):
    """Show results page for completed task"""
    if task_id not in processing_status or processing_status[task_id]["status"] != "complete":
        return templates.TemplateResponse("error.html", {"request": request, "message": "Processing not complete or task not found"})
    
    task_data = processing_status[task_id]
    
    # Get the original video file path and convert to web path
    file_path = task_data.get("file_path", "")
    if file_path:
        # Convert to web path
        web_file_path = "/" + file_path if not file_path.startswith("/") else file_path
    else:
        web_file_path = ""
    
    # Check for detection video with improved verification
    # Проверяем оба возможных формата видео (mp4 и avi)
    video_paths = [
        ("static/detection_video.mp4", "/static/detection_video.mp4"),
        ("static/detection_video.avi", "/static/detection_video.avi")
    ]
    static_video_path = None
    detection_video_path = None
    has_detection_video = False
    
    # Найдем первый существующий файл видео
    for local_path, web_path in video_paths:
        if os.path.exists(local_path):
            static_video_path = local_path
            detection_video_path = web_path
            break
    
    # Если нашли видео файл, проверяем его валидность
    if static_video_path:
        video_size = os.path.getsize(static_video_path)
        if video_size > 10000:  # Файл должен быть значительного размера (более 10KB)
            # Проверка валидности видео файла
            try:
                import cv2
                video = cv2.VideoCapture(static_video_path)
                if video.isOpened():
                    # Файл открывается и это действительно видео
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count > 0:
                        has_detection_video = True
                        print(f"Validated detection video: {static_video_path}, frames: {frame_count}")
                    else:
                        print(f"Warning: Video has no frames: {static_video_path}")
                    video.release()
                else:
                    print(f"Warning: Cannot open video: {static_video_path}")
            except Exception as e:
                print(f"Error validating video: {e}")
                # Если проверка не удалась, все равно пробуем показать видео
                has_detection_video = video_size > 100000  # Если файл больше 100KB, вероятно это видео
                print(f"Fallback to size check: {has_detection_video} (size: {video_size} bytes)")
        else:
            print(f"Warning: Video file too small: {video_size} bytes")
    else:
        print(f"Warning: Detection video not found in any format")
    
    # Добавляем текущее время в URL видео для предотвращения кэширования
    timestamp = int(time.time() * 1000)
    detection_video_url = f"{detection_video_path}?t={timestamp}" if has_detection_video else None
    
    # Если видео не найдено, но статус задачи успешный, обновим сообщение
    message = None
    if not has_detection_video and task_data.get("status") == "complete":
        message = "Видео результатов не найдено. Возможно, оно не было создано из-за ошибки или отсутствия детекций."
    
    return templates.TemplateResponse(
        "results.html", 
        {
            "request": request,
            "results": task_data.get("results", []),
            "stats": task_data.get("stats", {}),
            "video_path": web_file_path,
            "detection_video": detection_video_url,
            "timestamp": timestamp,
            "message": message
        }
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 