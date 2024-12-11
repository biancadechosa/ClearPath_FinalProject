from django.shortcuts import render
from django.http import JsonResponse
import torch
import cv2
import base64
import numpy as np
import json  

yolov5_repo_path = 'C:/Users/Leemart/ClearPath_FinalProject/ClearPath_System/yolov5'  
model_path = 'C:/Users/Leemart/ClearPath_FinalProject/ClearPath_System/object_detection/yolov5_model/best.pt' 

try:
    model = torch.hub.load(
        yolov5_repo_path,  
        'custom',           
        path=model_path,   
        source='local',     
        force_reload=True   
    )
except Exception as e:
    model = None
    error_message = f"Failed to load model: {e}"

def preprocess_image(image, size=(640, 640)):
    """
    Preprocess the image to match the input size and format expected by YOLOv5.
    """
    # Resize the image to the target size
    image_resized = cv2.resize(image, size)
    # Convert the image to RGB if it's in BGR format
    if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    return image_resized

def index(request):
   
    if model is None:
       
        return render(request, 'object_detection/index.html', {'error': error_message})

    if request.method == 'POST':
        cap = cv2.VideoCapture(0)  # Open default camera

        if not cap.isOpened():
            
            return render(request, 'object_detection/index.html', {'error': 'Unable to access the camera'})

        try:
            ret, frame = cap.read()
            if ret:
                # Preprocess the captured frame
                preprocessed_frame = preprocess_image(frame)

                # Run detection on the preprocessed frame
                results = model(preprocessed_frame)

                # Extract detections using pandas
                detections = results.pandas().xyxy[0]  # Pandas DataFrame with detections
                object_data = []

                for _, row in detections.iterrows():
                    class_name = row['name']
                    confidence = row['confidence']
                    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                    object_data.append(f"Detected {class_name} with confidence {confidence:.2f}")

                    # Draw bounding boxes on the original frame
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ({confidence:.2f})", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                img_bytes = buffer.tobytes()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                # Render the processed frame with detected objects
                return render(request, 'object_detection/index.html', {
                    'img_base64': img_base64,
                    'object_data': object_data  # Pass the detected object information
                })
            else:
                # Handle cases where no frame was read
                return render(request, 'object_detection/index.html', {'error': 'Unable to capture a frame from the camera'})
        finally:
            cap.release() 

    return render(request, 'object_detection/index.html')

def detect_objects(request):
    """
    API view to handle object detection for a POST request with a base64-encoded image.
    """
    if request.method == 'POST':
        try:
            # Extract the base64 image from the request
            data = json.loads(request.body)
            base64_image = data['image']
            
            # Decode the image and preprocess
            image_data = base64.b64decode(base64_image.split(',')[1])
            np_image = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            
            # Run detection with YOLO model
            results = model(img)
            detections = results.pandas().xyxy[0].to_dict(orient="records")
            
            # Return detections as JSON
            return JsonResponse({'objects': detections}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)
