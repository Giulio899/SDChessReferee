import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.hand=False

    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            self.hand=True
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        elif self.hand==True:
            print("mossa fatta")
            self.hand=False
            return img,True
        return img,False        

    def detect(self):
        cap = cv2.VideoCapture(0)
        flag=False
        while not flag:
            _,imgx = cap.read()
            img=cv2.flip(imgx,1) #flipped image
            img,flag = self.findHands(img)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
