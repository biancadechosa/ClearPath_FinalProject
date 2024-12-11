from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import torch
from flask_cors import CORS  

app = Flask(__name__)

CORS(app)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  

@app.route('/detect', methods=['POST'])
def detect():
    
    data = request.get_json()
    image_data = data['image']
    
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))

    results = model(img)  

    # Extract detected object classes
    detected_objects = results.names  # YOLOv5 object class names
    object_indices = results.xywh[0][:, -1].numpy()  # Object indices in the result
    objects_in_image = [detected_objects[int(idx)] for idx in object_indices]

    return jsonify({'objects': objects_in_image})

if __name__ == "__main__":
    app.run(debug=True, port=5000)  
