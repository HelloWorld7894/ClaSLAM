import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detectFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in results:
        #check if face has lower brightness than whole frame
        img_median = np.median(gray)
        face_median = np.median(gray[y:y+h, x:x+w])

        print(img_median)
        print(face_median)

        if face_median < img_median:
            radius = int(round((w + h)*0.25))
            center = (x + w//2, y + h//2)

            B_med, G_med, R_med = np.median(img[:, :, 0]), np.median(img[:, :, 1]), np.median(img[:, :, 2])
            cv2.circle(img, center, radius, (B_med, G_med, R_med), -1)

    return img