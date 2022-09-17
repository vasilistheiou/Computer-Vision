import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


cap = cv2.VideoCapture('PoseVideos1.mp4')
pTime = 0

while True:
    success, img = cap.read() #imge here is bgr but mp use rgb so we need to convert it
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversion
    results = pose.process(imgRGB) #send image to the model
    #we print all the landmarks but e dont know which is which. We cn go to mediapipe website and see wht landmarks they give
    #we can also place them int olists for further use.
    print(results.pose_landmarks)
    #draw landmarks
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            #to take the exat picture value
            cx, cy = int(lm.x*w) , int(lm.y*h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)




    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime


    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(10)