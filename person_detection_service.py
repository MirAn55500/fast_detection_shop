import cv2
import numpy as np
import pandas as pd
from threading import Thread, Event
from queue import Queue, Empty
import time
import os
import urllib.request
import shutil

class ObjectDetectionService:
    def __init__(self, use_custom_model=False):
        # Model configuration
        self.use_custom_model = use_custom_model
        self.model_files = self._download_model(use_custom_model)
        self.config_path = self.model_files['config']
        self.weights_path = self.model_files['weights']
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (416, 416)
        
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load appropriate labels
        self.labels = self._load_labels(use_custom_model)
        
        # Events of interest (classes)
        if use_custom_model:
            # Custom model with person, pallet, and dome classes
            self.target_classes = {
                'person': 0,  # Person class (index 0)
                'pallet': 1,  # Pallet class (index 1)
                'dome': 2,    # Dome class (index 2)
            }
        else:
            # Standard COCO dataset classes
            self.target_classes = {
                'person': 0,  # Person class in COCO (index 0)
                # Note: Pallets and domes aren't in COCO dataset
                # We'll need to train a custom model for those
            }
        
        # Threading and processing setup
        self.frame_queue = Queue(maxsize=30)
        self.detections = []
        self.processing_thread = None
        self.detection_thread = None
        self.stop_event = Event()
        self.total_frames = 0
        self.processed_frames = 0
        self.last_save_time = time.time()
        self.output_file = 'detection_results.xlsx'
        self.last_frame_time = 0
        self.video_writer = None
        self.video_fps = 0
        self.video_size = None
        self.event_frames_dir = 'static/event_frames'
        self.file_path = None
        self.detection_video_path = 'static/detection_video.mp4'
        
        # Progress callback
        self.progress_callback = None
        
        print("Object Detection Service initialized")
        print(f"Using {'custom' if use_custom_model else 'standard'} model")
        print(f"Model input size: {self.input_size}")
        print(f"Confidence threshold: {self.confidence_threshold}")

    def _download_model(self, use_custom_model=False):
        """Download model files if not already present"""
        model_dir = os.path.join(os.getcwd(), 'models')
        
        if use_custom_model:
            # Custom model paths
            config_path = os.path.join(model_dir, 'yolov4-tiny-custom.cfg')
            weights_path = os.path.join(model_dir, 'yolov4-tiny-custom.weights')
            
            # Check if custom model exists
            if not (os.path.exists(config_path) and os.path.exists(weights_path)):
                print("Custom model files not found. Please train the model first using custom_model_training.py")
                print("Falling back to standard YOLO model...")
                return self._download_model(use_custom_model=False)
            
            print(f"Using custom model from {model_dir}")
            return {'config': config_path, 'weights': weights_path}
        else:
            # Standard YOLO model paths
            config_path = os.path.join(model_dir, 'yolov4-tiny.cfg')
            weights_path = os.path.join(model_dir, 'yolov4-tiny.weights')
            
            # Create models directory if it doesn't exist
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Download config file if it doesn't exist
            if not os.path.exists(config_path):
                print("Downloading YOLO-Tiny V4 configuration file...")
                
                # URL for config file
                config_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
                
                try:
                    # Add headers to mimic a browser request
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                    }
                    
                    # Download config file
                    print(f"Downloading config from: {config_url}")
                    req = urllib.request.Request(config_url, headers=headers)
                    with urllib.request.urlopen(req) as response, open(config_path, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)
                    print(f"Config file saved to {config_path}")
                    
                except Exception as e:
                    print(f"Error downloading config file: {e}")
                    print("\nPlease download the config file manually:")
                    print(f"1. Download config file from {config_url}")
                    print(f"2. Save it to {config_path}")
                    raise Exception("Failed to download config file automatically. Please download it manually.")
            
            if not os.path.exists(weights_path):
                print("Downloading YOLO-Tiny V4 weights file...")
                
                weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
                
                try:
                    # Add headers to mimic a browser request
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                    }
                    
                    # Download weights file
                    print(f"Downloading weights from: {weights_url}")
                    print("This may take a while...")
                    req = urllib.request.Request(weights_url, headers=headers)
                    with urllib.request.urlopen(req) as response, open(weights_path, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)
                    print(f"Weights file saved to {weights_path}")
                    
                    print("Model files ready")
                    
                except Exception as e:
                    print(f"Error downloading weights file: {e}")
                    print("\nPlease download the weights file manually:")
                    print(f"1. Download weights file from {weights_url}")
                    print(f"2. Save it to {weights_path}")
                    raise Exception("Failed to download weights file automatically. Please download it manually.")
            else:
                print(f"Using existing model files at {model_dir}")
            
            return {'config': config_path, 'weights': weights_path}

    def _load_labels(self, use_custom_model=False):
        """Load the labels for the model"""
        model_dir = os.path.join(os.getcwd(), 'models')
        
        if use_custom_model:
            # Custom model labels
            labels_path = os.path.join(model_dir, 'custom.names')
            if not os.path.exists(labels_path):
                print("Custom labels file not found. Creating default custom labels...")
                with open(labels_path, 'w') as f:
                    f.write("person\npallet\ndome\n")
        else:
            # COCO dataset labels
            labels_path = os.path.join(model_dir, 'coco.names')
            
            # Create labels file if it doesn't exist
            if not os.path.exists(labels_path):
                # COCO dataset labels (80 classes for YOLO)
                coco_labels = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
                
                # Write the labels to a file
                with open(labels_path, 'w') as f:
                    for label in coco_labels:
                        f.write(f"{label}\n")
        
        # Load the labels
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        
        return labels

    def _detect_objects(self, image):
        """Detect objects in the image using YOLO model"""
        # Get image dimensions
        height, width, _ = image.shape
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, self.input_size, swapRB=True, crop=False)
        
        # Set the input to the network
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected objects
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in outputs:
            # Process each detection
            for detection in output:
                # The first 4 values are box coordinates, the rest are class scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter out weak predictions
                if confidence > self.confidence_threshold:
                    # YOLO returns the center (x, y) and width/height
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Add to lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Process the results
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]
                
                # Ensure class_id is within range of labels
                if class_id < len(self.labels):
                    class_name = self.labels[class_id]
                    
                    # Filter to include only people when using standard model
                    if not self.use_custom_model and class_name != 'person':
                        continue
                    
                    # Ensure box coordinates are within image boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(width - x, w)
                    h = min(height - y, h)
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'box': [x, y, x + w, y + h]  # [x1, y1, x2, y2] format
                    })
        
        return detections

    def _draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image"""
        annotated_image = image.copy()
        
        for detection in detections:
            # Get the bounding box
            x1, y1, x2, y2 = detection['box']
            
            # Draw the bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_image

    def process_frames(self, video_path):
        """Process video frames using OpenCV"""
        try:
            # Save the file path
            self.file_path = video_path
            
            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}")
                self.frame_queue.put(None)
                return
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                self.frame_queue.put(None)
                return
            
            # Get video properties
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_size = (width, height)
            
            # Calculate duration
            duration_s = self.total_frames / self.video_fps if self.video_fps > 0 else 0
            
            # Process every 16th frame (reduced as requested)
            frame_interval = 16
            
            print(f"Video Details - Size: {self.video_size}, FPS: {self.video_fps:.2f}")
            print(f"Duration: {duration_s:.2f}s, Total Frames: {self.total_frames}")
            print(f"Processing 1 frame every {frame_interval} frames (approx. {self.video_fps/frame_interval:.1f} FPS)")
            
            # Ensure static directory exists
            os.makedirs('static', exist_ok=True)
            
            # Initialize video writer for annotated output
            if self.video_size and self.video_size[0] > 0 and self.video_size[1] > 0 and self.video_fps > 0:
                # Use temp file first to ensure complete writing
                temp_video_path = 'static/temp_detection_video.mp4'
                
                # Попробуем использовать кодек H.264, который лучше поддерживается в браузерах
                try:
                    # Сначала пробуем использовать кодек H.264
                    # Для Windows часто нужно использовать XVID или DIVX вместо H264
                    # Для Linux/Mac можно использовать mp4v или avc1
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    self.video_writer = cv2.VideoWriter(
                        temp_video_path,
                        fourcc,
                        self.video_fps / frame_interval,  # Output at processed FPS
                        self.video_size
                    )
                    
                    # Проверим, что writer инициализирован успешно
                    if not self.video_writer.isOpened():
                        raise Exception("Failed to initialize VideoWriter with avc1 codec")
                    
                    print(f"VideoWriter initialized with avc1 codec")
                    
                except Exception as e:
                    print(f"Error with avc1 codec: {e}, trying mp4v...")
                    try:
                        # Если не получилось с H.264, пробуем mp4v
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(
                            temp_video_path,
                            fourcc,
                            self.video_fps / frame_interval,
                            self.video_size
                        )
                        
                        if not self.video_writer.isOpened():
                            raise Exception("Failed to initialize VideoWriter with mp4v codec")
                            
                        print(f"VideoWriter initialized with mp4v codec")
                        
                    except Exception as e2:
                        print(f"Error with mp4v codec: {e2}, trying XVID...")
                        # Последняя попытка с XVID (широко поддерживается)
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            self.video_writer = cv2.VideoWriter(
                                temp_video_path.replace('.mp4', '.avi'),  # меняем расширение на .avi для XVID
                                fourcc,
                                self.video_fps / frame_interval,
                                self.video_size
                            )
                            
                            if not self.video_writer.isOpened():
                                raise Exception("Failed to initialize VideoWriter with XVID codec")
                                
                            # Обновляем путь для соответствия формату AVI
                            self.detection_video_path = 'static/detection_video.avi'
                            print(f"VideoWriter initialized with XVID codec (AVI format)")
                            
                        except Exception as e3:
                            print(f"Error initializing any video codec: {e3}")
                            self.video_writer = None
            
            frame_count = 0
            processed_frame_count = 0
            
            # Process frames
            while cap.isOpened() and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached.")
                    break
                
                frame_count += 1
                
                # Calculate current time in video
                current_time = frame_count / self.video_fps if self.video_fps > 0 else 0
                
                # Update progress through callback if provided
                if self.progress_callback and frame_count % 30 == 0:
                    # Calculate progress based on total frames
                    progress_percent = min(int((frame_count / self.total_frames) * 100), 100)
                    self.progress_callback(
                        progress_percent, 
                        100,
                        f"Processing frame {frame_count}/{self.total_frames}"
                    )
                
                # Process only every Nth frame (every 8th frame as requested)
                if frame_count % frame_interval == 0:
                    try:
                        # Put frame in queue for detection thread
                        self.frame_queue.put((frame.copy(), current_time), timeout=1)
                        processed_frame_count += 1
                        self.processed_frames += 1
                        
                        # Show progress
                        if processed_frame_count % 5 == 0:
                            progress = (frame_count / self.total_frames) * 100
                            print(f"Progress: {progress:.1f}% ({frame_count}/{self.total_frames} frames)")
                    except Empty:
                        print(f"Queue full. Detection might be slow.")
                        pass
                    except Exception as e:
                        print(f"Error putting frame in queue: {e}")
                        continue
            
        except Exception as e:
            print(f"Error in process_frames: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
                print("Video capture released.")
            self.frame_queue.put(None)
            print("Frame processing finished.")

    def detect_objects_thread(self):
        """Thread for object detection"""
        print("Starting object detection thread")
        os.makedirs(self.event_frames_dir, exist_ok=True)
        
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=1)
                if frame_data is None:
                    break
                    
                frame, timestamp = frame_data
                
                # Detect objects in the frame
                detections = self._detect_objects(frame)
                
                # Draw detections on the frame
                annotated_frame = self._draw_detections(frame, detections)
                
                # Save annotated frame to video
                if self.video_writer is not None:
                    try:
                        self.video_writer.write(annotated_frame)
                    except Exception as e:
                        print(f"Error writing frame to video: {e}")
                
                # Process detections and record events
                for detection in detections:
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    # Check if this is a target class we're interested in
                    is_target = False
                    event_type = None
                    
                    if class_name == 'person':
                        is_target = True
                        event_type = "Person detected"
                    elif class_name == 'pallet' and self.use_custom_model:
                        is_target = True
                        event_type = "Pallet detected"
                    elif class_name == 'dome' and self.use_custom_model:
                        is_target = True
                        event_type = "Dome detected"
                    
                    if is_target:
                        # Format timestamp for display (используем UTC для независимости от часового пояса)
                        if timestamp > 0:
                            # Преобразуем timestamp в секундах в относительное время видео
                            hours = int(timestamp // 3600)
                            minutes = int((timestamp % 3600) // 60)
                            seconds = int(timestamp % 60)
                            # Форматируем как "ЧЧ:ММ:СС"
                            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        else:
                            time_str = f"F:{self.processed_frames}"
                        
                        # Save frame for this event with a unique filename
                        event_frame_filename = f"event_{len(self.detections):04d}.jpg"
                        event_frame_path = os.path.join(self.event_frames_dir, event_frame_filename)
                        cv2.imwrite(event_frame_path, annotated_frame)
                        
                        # Record the detection with web-friendly path
                        self.detections.append({
                            'timestamp': time_str,
                            'event_type': event_type,
                            'class': class_name,
                            'confidence': confidence,
                            'image_path': f"/static/event_frames/{event_frame_filename}"
                        })
                
                # Save intermediate results periodically
                current_time = time.time()
                if current_time - self.last_save_time >= 10:
                    self.save_intermediate_results()
                    self.last_save_time = current_time
                    
            except Empty:
                continue
            except Exception as e:
                print(f"Error in detect_objects_thread: {e}")
                import traceback
                traceback.print_exc()
                continue

    def save_intermediate_results(self):
        """Save detection results to Excel file"""
        if self.detections:
            # Create a copy of detections for saving
            detections_to_save = list(self.detections)
            if detections_to_save:
                df = pd.DataFrame(detections_to_save)
                if not df.empty:
                    try:
                        # Reorder columns to match required format
                        columns = ['timestamp', 'event_type', 'image_path', 'class', 'confidence']
                        df = df[columns]
                        df.to_excel(self.output_file, index=False, engine='openpyxl')
                        print(f"Saved {len(detections_to_save)} events to {self.output_file}")
                    except Exception as e:
                        print(f"Error saving to Excel: {e}")

    def process_video(self, video_path):
        """Process a video file"""
        try:
            # Reset state for new video
            self.detections = []
            self.processed_frames = 0
            self.last_save_time = time.time()
            self.stop_event.clear()
            self.file_path = video_path

            # Start processing threads
            self.processing_thread = Thread(target=self.process_frames, args=(video_path,))
            self.detection_thread = Thread(target=self.detect_objects_thread)
            
            self.processing_thread.start()
            self.detection_thread.start()
            
            # Wait for threads to complete
            self.processing_thread.join()
            self.detection_thread.join()
            
            # Final save
            print("\nSaving final results...")
            self.save_intermediate_results()
            
            # Clean up video writer and ensure file is properly saved
            has_detection_video = False
            if self.video_writer is not None:
                try:
                    # Убедимся, что все фреймы записаны
                    self.video_writer.release()
                    self.video_writer = None
                    print("Video writer released")
                    
                    # Move temp video to final location
                    temp_video_path = 'static/temp_detection_video.mp4'
                    if os.path.exists(temp_video_path):
                        # Check if file is valid and has size > 0
                        if os.path.getsize(temp_video_path) > 0:
                            # Перед перемещением проверяем наличие и удаляем старый файл
                            if os.path.exists(self.detection_video_path):
                                try:
                                    os.remove(self.detection_video_path)
                                    print(f"Removed existing detection video")
                                except Exception as e:
                                    print(f"Error removing existing video: {e}")
                            
                            # Перемещаем временный файл в конечное расположение
                            try:
                                shutil.move(temp_video_path, self.detection_video_path)
                                print(f"Moved temp video to {self.detection_video_path}")
                            except Exception as e:
                                print(f"Error moving video file: {e}")
                                # Если перемещение не удалось, пробуем скопировать и удалить
                                try:
                                    shutil.copy2(temp_video_path, self.detection_video_path)
                                    os.remove(temp_video_path)
                                    print(f"Copied temp video to final location")
                                except Exception as copy_error:
                                    print(f"Error copying video file: {copy_error}")
                            
                            # Проверяем результат
                            has_detection_video = os.path.exists(self.detection_video_path)
                            if has_detection_video:
                                video_size = os.path.getsize(self.detection_video_path)
                                print(f"\nAnnotated video saved to {self.detection_video_path}")
                                print(f"Video size: {video_size} bytes")
                                
                                # Дополнительная проверка валидности видео
                                try:
                                    import cv2
                                    cap = cv2.VideoCapture(self.detection_video_path)
                                    if cap.isOpened():
                                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                        if frame_count > 0:
                                            print(f"Video validation successful: {frame_count} frames")
                                        else:
                                            print("Warning: Video has no frames")
                                            has_detection_video = False
                                        cap.release()
                                    else:
                                        print("Warning: Created video cannot be opened")
                                        has_detection_video = False
                                except Exception as e:
                                    print(f"Error validating video: {e}")
                            else:
                                print(f"Error: Failed to move temp video to {self.detection_video_path}")
                        else:
                            print("Warning: Temp video file has zero size")
                    else:
                        print(f"Warning: Temp video file not found at {temp_video_path}")
                        
                except Exception as e:
                    print(f"Error finalizing video: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Final callback to indicate completion
            if self.progress_callback:
                self.progress_callback(100, 100, "Processing complete")
            
            # Create final DataFrame
            final_detections = list(self.detections)
            if final_detections:
                df_report = pd.DataFrame(final_detections)
                if not df_report.empty:
                    # Reorder columns to match required format
                    columns = ['timestamp', 'event_type', 'image_path', 'class', 'confidence']
                    df_report = df_report[columns]
                return df_report
            else:
                return pd.DataFrame()

        except KeyboardInterrupt:
            print("\nInterrupted by user, shutting down...")
            self.stop_event.set()
            # Wait for threads to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=5)
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            # Save what we have
            self.save_intermediate_results()
            return pd.DataFrame(list(self.detections))
        except Exception as e:
            print(f"\nError in process_video: {e}")
            import traceback
            traceback.print_exc()
            self.stop_event.set()
            # Report error via callback
            if self.progress_callback:
                self.progress_callback(0, 100, f"Error: {str(e)}")
            return pd.DataFrame()
        finally:
            self.stop_event.set()
            # Ensure threads are joined
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2)
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=2)
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None


def main():
    """Main function"""
    print("Object Detection Service")
    print("1. Use standard model (person detection only)")
    print("2. Use custom model (person, pallet, dome detection)")
    
    choice = input("Select model (1/2): ")
    use_custom_model = (choice == "2")
    
    service = ObjectDetectionService(use_custom_model=use_custom_model)
    video_path = input("Enter path to video file: ")
    if not video_path:
        print("No video path provided. Exiting.")
        return

    start_time = time.time()
    results_df = pd.DataFrame()
    
    try:
        results_df = service.process_video(video_path)
        end_time = time.time()
        
        print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
        
        if results_df is not None and not results_df.empty:
            print(f"\nTotal events detected: {len(results_df)}")
            print("\nEvent type statistics:")
            if 'event_type' in results_df.columns:
                print(results_df['event_type'].value_counts())
            print("\nSample results:")
            print(results_df.head())
        else:
            print("No events detected or processing failed.")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down service...")
        service.stop_event.set()

if __name__ == "__main__":
    main() 