import mediapipe as mp
import time
import cv2

cap = cv2.VideoCapture("FaceVideo.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    #print(results.detections)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #print(id, detection) #label id is the id of the face
            mpDraw.draw_detection(img,detection)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'Fps: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(10)

