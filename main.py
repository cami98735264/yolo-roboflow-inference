import base64
from PIL import Image
import cv2
import os
import pytesseract
import numpy as np
from roboflow import Roboflow
os.environ["TESSDATA_PREFIX"] = "C:/Program Files/Tesseract-OCR/tessdata"
rf = Roboflow(api_key="4Vkwb5mkP0K6pBH1xQoN")
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(4).model

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()
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
            cv2.imwrite("idk.jpg", ocr_frame)
            gray = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2GRAY)
            reduc_noise = cv2.bilateralFilter(gray, 11, 17, 17) 
            img_filtered = Image.fromarray(reduc_noise)
            numpy_img = np.asarray(img_filtered)
            cv2.imwrite("idk.jpg", numpy_img)
            predicted_result = pytesseract.image_to_string(img_filtered, config ='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNÑOPQRSTUVWXYZ0123456789')
            ocred = "".join(predicted_result.split()).replace(":", "").replace("-", "")
            print(ocred)


    # Display the frame
    # Resize cv2 window to 640x480
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.resize(frame, (640, 480))
    cv2.imshow('Camera', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()