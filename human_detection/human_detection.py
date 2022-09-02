import cv2
import time
import numpy as np
import pathlib

if __name__ == '__main__':
    ESC_KEY = 27
    INTERVAL = 33
    FRAME_RATE = 30
    cycle = 30

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = "/home/jet/workspace/human_detection/MyVideo_2_Trim.mp4"

    cascade_file = "/home/jet/workspace/human_detection/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_file)


    cap = cv2.VideoCapture(DEVICE_ID)


    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape


    print(cap.get(cv2.CAP_PROP_FPS))
    n = 0


    while end_flag == True:
        start = time.time()
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100,100))
        FACES = np.array(face_list)

        if (FACES.size > 0) and n >= cycle:
            n = 0
            file = pathlib.Path('trigger.txt')
            file.touch()
            cv2.imwrite('photo.jpg', c_frame)
            print("human_detection")

        n += 1
        end_time = time.time() - start
        print("1frametimetime:{0}".format(end_time)+"[sec]")

        end_flag, c_frame = cap.read()

    cap.release()



