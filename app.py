from flask import Flask,render_template,Response
import base64
import os
from dotenv import load_dotenv
load_dotenv() 
api_key = os.getenv("API_KEY")
open_alpr_path = os.getenv("OPEN_ALPR_PATH")
from PIL import Image
os.add_dll_directory(open_alpr_path)
import json
from openalpr import Alpr
import cv2
import numpy as np
from roboflow import Roboflow
os.environ["TESSDATA_PREFIX"] = "C:/Program Files/Tesseract-OCR/tessdata"
rf = Roboflow(api_key="4Vkwb5mkP0K6pBH1xQoN")
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(4).model

# Initialize OpenALPR
alpr = Alpr("us", open_alpr_path + "openalpr.conf", open_alpr_path + "runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    sys.exit(1)

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
       # Read the frame from the camera
       ret, frame = camera.read()
       height, width, channels = frame.shape
       scale = 640 / max(height, width)
       frame = cv2.resize(frame, (round(scale * width), round(scale * height)))
       # Open frame with Image.open()
       img = Image.fromarray(frame)
       numpydata = np.asarray(img)
       if not ret:
        print("Unable to capture video")
        break

       results = model.predict(numpydata, confidence=40, overlap=30).json()
       if results["predictions"]:
        for prediction in results["predictions"]:
            x0 = prediction['x'] - prediction['width'] / 2
            x1 = prediction['x'] + prediction['width'] / 2
            y1 = prediction['y'] + prediction['height'] / 2
            y0 = prediction['y'] - prediction['height'] / 2
            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            ocr_frame = frame[int(y0):int(y1), int(x0):int(x1)]
            temp_img = cv2.imwrite("temp.jpg", ocr_frame)
            results = alpr.recognize_file("./temp.jpg")
            if results['results']:
                print(f"Recognized plate: {results['results'][0]['plate']}, with a confidence of {results['results'][0]['confidence']}%")
            ret, buffer = cv2.imencode('.jpg', frame) 
            jpg_frame = buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')