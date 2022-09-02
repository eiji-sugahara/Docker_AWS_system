import cv2

filepath = "/home/jet/workspace/human_detection/MyVideo_2_Trim.mp4"
cap = cv2.VideoCapture(filepath)

avg = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    cv2.accumulateWeighted(gray, avg, 0.6)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    #print(frameDelta)
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    #print(thresh)

cap.release()


