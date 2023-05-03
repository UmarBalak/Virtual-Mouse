import cv2
import time
import hand_tracking as ht
import numpy as np
import autopy

from pynput.mouse import Button, Controller
mouse = Controller()


smoothening = 5
wCam, hCam = 640, 460
frameR = 120 # frame red]uction
wScr, hScr = autopy.screen.size()
print(wScr, hScr)

pTime = 0
pLocX, pLocY = 0, 0 # current location, previous location
cLocX, cLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3,wCam) # 3 is default id for width
cap.set(4,hCam) # 4 is default id for height
detector = ht.handDetector(maxHands=1)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 2)

        # moving mode
        if (fingers[1] == 1 and fingers[2] == 1) and (fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0):
            distance = detector.findDistance(8, 12, img)
            # print(distance)
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening

            if distance < 40:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cv2.circle(img, (cx,cy), 10, (0,0,255), cv2.FILLED)
                autopy.mouse.move(wScr - cLocX, cLocY)
                pLocX, pLocY = cLocX, cLocY

            elif distance > 40:
                mouse.click(Button.left, 1)
                time.sleep(1)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(40,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)


    cv2.imshow('Image',img)
    cv2.waitKey(1)