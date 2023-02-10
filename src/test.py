import cv2

vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    c

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break