import cv2
import numpy as np
import torch
from torchvision.models.detection import *

#face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#pytorch
CLASSES = 71 #we want to detect human being
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = fasterrcnn_mobilenet_v3_large_320_fpn(weights=True, progress=True, pretrained_backbone=True).to(DEVICE)
MODEL.eval()

"""
def detectFace(img, img_cluster):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_cluster = cv2.cvtColor(img_cluster, cv2.COLOR_BGR2GRAY)

    results = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in results:

        radius = int(round((w + h)*0.25))
        center = (x + w//2, y + h//2)

        B_med, G_med, R_med = np.median(img_cluster[:, :, 0]), np.median(img_cluster[:, :, 1]), np.median(img_cluster[:, :, 2])
        cv2.circle(img_cluster, center, radius, (B_med, G_med, R_med), -1)

    return img_cluster
"""

def DetectPytorch(img):
    img_orig = img.copy()

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    image = image.to(DEVICE)
    detections = MODEL(image)[0]

    startX, startY, endX, endY = None, None, None, None

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i] * 100
        idx = int(detections["labels"][i])

        if confidence > 85:
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{:.2f}%".format(confidence)
            print("[INFO] {}".format(label))
            print("[DVEŘE?]: " + str(idx))
    
    cv2.rectangle(img_orig, (startX, startY), (endX, endY), (0, 0, 255), 3)
    return img_orig

vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    img_nn = DetectPytorch(frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    cv2.imshow("frame", img_nn)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break