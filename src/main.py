import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import math

cap = cv2.VideoCapture(0)

def Thresh(img):
    thresh = 50
    img[img > thresh] = 255
    img[img <= thresh] = 0

    return img

def HoughLines(img_in, img_out, mode):
    if mode == "std":
        lines = cv2.HoughLines(img_in, 1, np.pi / 180, 150, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(img_out, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    elif mode == "p":
        linesP = cv2.HoughLinesP(img_in,rho = 1,theta = 1*np.pi/180,threshold = 40,minLineLength = 60,maxLineGap = 10)
    
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(img_out, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)



while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    test = img.copy()

    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #dst = cv2.Canny(hpf, 30, 150, None, 3)
    #print(dst.dtype)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    dst = Thresh(dst)
    #dst = cv2.blur(dst,(5,5))
    #dst = Thresh(dst)

    HoughLines(dst, test, "p")
            
    cv2.imshow("Source", img)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", test)
    cv2.imshow("sobel", dst)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
