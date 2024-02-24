import cv2
import numpy as np

# ------------------- initializing paths for mobile net ssd model------------------- #
protox = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"

# --------------------defining minimum confidence level for detection--------------- #
min_confidence = 0.2

#  ------------------ classes in the model------------------------------------------ #
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

#  ----------------------------to count number of frames------------------------------ #
frame_no =0

#---------------------------capturing the video--------------------------------------- #
cap = cv2.VideoCapture("enter ip address of camera")
net = cv2.dnn.readNetFromCaffe(protox, model)

#  ---------------------------to keep track track of the centroids-------------------- #
centr_pts_pre_fr = []
# --------------------------processing of frames-------------------------------------- #

while (True):

    # -----capturing of frames -------#
    ret,frame = cap.read()
    if not ret:
        break
    
    frame_no +=1
    # ----- fran=mes resizing----------#
    frame = cv2.resize(frame , (1000,600))
    # ------ list to store centroids of the cuirrent fraame--------#
    centr_pts_cur_fr = []

    (h,w) = frame.shape[:2]
    print("height:",h, "width:",w)

    # --------blob that is feeded in the deep nueral network----------#
    blob = cv2.dnn.blobFromImage(cv2.resize(frame , (400,400)), 0.007843,(300,300),127.5)
    net.setInput(blob)
    detect = net.forward()
    # print("this is detections")
    # print (detect)

    # ------- loop for object detection---------#
    for i in np.arange(0,detect.shape[2]):
        confidence = detect[0,0,i,2]
        if confidence >= min_confidence:
            idx = detect[0,0,i,1]
            if classes[int(idx)] != "person":
                continue


            bboxes = detect[0,0,i,3:7] * np.array([w,h,w,h])
            (x1,y1,x2,y2) = bboxes.astype("int")

            cx = int((x1+ x2)/2)
            cy = int((y1 + y2)/2)

            centr_pts_cur_fr.append((cx,cy))

            print( "start x:",x1,"start y:",y1,"end x:",x2, "end y :",y2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

            cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

            print("current frame :")
            print(centr_pts_cur_fr)

            print("previous frame :")
            print(centr_pts_pre_fr)

            centr_pts_pre_fr = centr_pts_cur_fr.copy()

    head_count =len(centr_pts_cur_fr)

    info = [("TOTAL VISIBLE PEOPLE",head_count)]

    for (x,y) in info:
        text = "{}: {}".format(x, y)
        cv2.putText(frame, text, (50,30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()                