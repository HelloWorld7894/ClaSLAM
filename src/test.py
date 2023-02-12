import cv2

vid = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while(True):
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in results:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        radius = int(round((w + h)*0.25))
        center = (x + w//2, y + h//2)

        frame = cv2.circle(gray, center, radius, 50, -1)

        #frame[y:y+h, x:x+w] = 0

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break