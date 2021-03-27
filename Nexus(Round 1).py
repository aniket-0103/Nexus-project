import cv2
import numpy as np
import time

# Function to detect the nearest shape and find the color of the detected shape
def detect_the_color(img, contour):
    y_val = []
    x_val = []
    conts = []

    for cnt in contour:
        area = cv2.contourArea(cnt)
        if (area > 4000 and area < 20000):

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                y_val = y_val + [cY]
                x_val = x_val + [cX]
                conts = conts + [cnt]
            else:
                cX = 0
                cY = 0
    if len(y_val)!=0:
        ind = y_val.index(max(y_val))  # Y value of the center pixel of the nearest shape

        cv2.drawContours(img, conts, ind, (0, 0, 255), 5)
        X = x_val[ind]
        Y = y_val[ind]

        approx = cv2.approxPolyDP(conts[ind], 0.01 * cv2.arcLength(conts[ind], True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        c = img[Y,X]

        # if c[0]>c[1] and c[0]>c[2]:
        #     cv2.putText(img, "Blue circle", (x-10,y-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0),2)
        #     color = "blue"
        #
        # elif c[0] < c[1] and c[1] > c[2]:
        #     cv2.putText(img, "Green circle", (x-5, y-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0),2)
        #     color = "green"
        #
        # elif c[0] < c[1] and c[1] > c[2]:
        #     cv2.putText(img, "Red circle", (x-10, y-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0),2)
        #     color = "green"

        if len(approx==3):
            shape = "Triangle"
            cv2.putText(img, "Triangle", (x - 10, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0), 2)

        elif (len(approx)==4):
            shape = "Rectangle"
            cv2.putText(img, "Rectangle", (x - 10, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0), 2)

        else :
            shape = "Circle"
            cv2.putText(img, "Circle", (x - 10, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0), 2)

        return X,Y, shape
    else: return 0,0,None

cap= cv2.VideoCapture(0)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while (True):

    ret, frame = cap.read()

    if frame is not None:
        # Canny edge detection
        imgc = frame
        imgc = cv2.GaussianBlur(imgc, (7, 7), 0)
        imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
        imgc = cv2.Canny(imgc, 100, 40)

        kernel = np.ones((5, 5))
        imgc = cv2.dilate(imgc, kernel, 0)

        contours, _ = cv2.findContours(imgc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cX,cY,shape = detect_the_color(frame,contours)

        if shape=="Triangle":

            if cX>150 and cX<200:
                print("forward")

            else :
                print("Left")

        elif shape=="Rectangle":

            if cX > 150 and cX < 200:
                print("Forward")

            else:
                print("Right")

        elif shape=="Circle":

            if cX > 150 and cX < 200:
                print("Forward")

            elif cX > 200:
                print("Right")

            else:
                print("Left")

    else: break
    time.sleep(0.05)
    cv2.imshow("frame",frame)
    if(cv2.waitKey(1)==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
