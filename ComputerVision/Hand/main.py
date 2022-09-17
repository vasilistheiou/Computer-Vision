import cv2
import mediapipe as mp #needs python 3.7.9
import time

from mediapipe.python.solutions.hands import Hands

cap = cv2.VideoCapture(0) #opens camera

mpHands = mp.solutions.hands
hands = mpHands.Hands() #its is an object
mpDraw = mp.solutions.drawing_utils
#for frame rate
pTime = 0
cTime = 0

while True:

  success, img = cap.read()
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert image to RGB
  results = hands.process(imgRGB)
  #print(results.multi_hand_landmarks) #to check if it detects something

  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks: #it is for every signle hand
      for id, lm in enumerate(handLms.landmark): #landmark = x,y,z coorinations
        print(id, lm) #they give ratio of image so
        h, w, c = img.shape #take width and height and multiply with width and height to take pixel value
        cx, cy = int(lm.x*w), int(lm.y*h) #position of the center for all values
        print(id, cx, cy)
        if id == 0: #we detect a specific "point" - part of the hand/ we can put point in a list and detect movement etc
          cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)

      mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #draws hand points and hand connections

  cTime = time.time() #give current time
  fps = 1/(cTime - pTime)
  pTime = cTime

  #display frame rate on screen
  cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3 , (255,0,255), 3)

  cv2.imshow("Image", img)
  cv2.waitKey(1)