import cv2
import numpy as np

#face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

"""
#pytorch
CLASSES = 1 #we want to detect human being
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = fasterrcnn_mobilenet_v3_large_320_fpn(weights=True, progress=True, pretrained_backbone=True).to(DEVICE)
MODEL.eval()


#mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
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
NOT REALLY A GOOD IDEA
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

        if idx == CLASSES and confidence > 90:
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{}: {:.2f}%".format(CLASSES, confidence)
            print("[INFO] {}".format(label))

            #check if face has lower brightness than whole frame
            gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

            img_median = np.median(gray)
            face_median = np.median(gray[startY:endY, startX:endX])

            print(img_median)
            print(face_median)

            if face_median < img_median:
                w = endX - startX
                h = endY - startY

                radius = int(round((w + h)*0.25))
                center = (startX + w//2, startY + h//2)

                B_med, G_med, R_med = np.median(img_orig[:, :, 0]), np.median(img_orig[:, :, 1]), np.median(img_orig[:, :, 2])
                cv2.circle(img_orig, center, radius, (B_med, G_med, R_med), -1)

    
    return img_orig
"""