# Importing the modules
import mediapipe as mp
import cv2
import time

#Creating a video capture instanc
cap = cv2.VideoCapture(0)
pTime = 0

# Creating a MediaPipe Face Detection processes an RGB -
# - image and returns a list of the detected face location data

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils # Creating a utilities class
faceDetection = mpFaceDetection.FaceDetection(0.75) #Initializing face detection Object

while True:
    success, img = cap.read() # Reading the image

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converting to rgb
    results = faceDetection.process(imgRGB) # Processing the image and getting list of faces detected
    print(results)

    if results.detections: #if any faces detected
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection) # Draw the boxes and landmarks for detected faces
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih) # its a tuple!
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]* 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)