import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        # importing media pipe classes and function
        self.mpFaceDetection = mp.solutions.face_detection # importing mediapipe face detection module
        self.mpDraw = mp.solutions.drawing_utils # Creating a utilities class
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon) #Initializing face detection Object

    def findFaces(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converting to rgb
        self.results = self.faceDetection.process(imgRGB) # Processing the image and getting list of faces detected
        # print(results) # printing a list of the detected face location data
        bboxs = []
        if self.results.detections: #if any faces detected
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box # relative values of bounding box
                ih, iw, ic = img.shape #Getting the shape of the processed frames
                # Calculating corner pixel values of bounding box
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # bbox a tuple!
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                cv2.putText(img, f'{int(detection.score[0]* 100)}%', (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) # Adding the detection score to each bounding box
                return img, bbox

    def fancyDraw(self, img, bbox, len = 30, thickness = 5, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 255), 2, rt)  # Drawing the bounding box
        #Top left
        cv2.line(img, (x, y), (x + len, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y + len), (255, 0, 255), thickness)

        #Top right
        cv2.line(img, (x1, y), (x1 - len, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + len), (255, 0, 255), thickness)

        #Bottom left
        cv2.line(img, (x, y1), (x + len, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - len), (255, 0, 255), thickness)

        #Bottom right
        cv2.line(img, (x1, y1), (x1 - len, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - len), (255, 0, 255), thickness)

        return img

def main():
    # Creating a video capture instanc
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()  # Reading the image and getting frames
        img, bbox = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)  # Sets the frame rate
        print(bbox)

if __name__ == "__main__":
    main()