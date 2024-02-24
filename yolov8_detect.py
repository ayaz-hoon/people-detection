import cv2
from ultralytics import YOLO
import numpy as np
# ------- capture the video frames -------------#
cap = cv2.VideoCapture("enter your cameras ip address")

# ---------- initializing yolo model-------------#
model = YOLO("yolov8m.pt")

while True:
    # ---------- capturing frames-----------#
    ret , frame = cap.read()
    if not ret :
        break
    # ---------- resizing the frames---------#
    frame = cv2.resize(frame , (1400,800))

    # --------- list that stores the centroids of the current frame---------#
    centr_pt_cur_fr = []

    results = model(frame)
    result = results[0]
    # print("this is results :")
    # print(results)
    print("this is shape of frame,",frame.shape)
    print("this is result :")
    print(result)

    # classes = np.array(result.boxes.names.cpu())
    # print("this is classes:",classes)

    # ------- to get the classes of the yolo model to filter out the people---------------#
    classes = np.array(result.boxes.cls.cpu(),dtype="int")
    print("this is classes:",classes)

    # ---------confidence level of detections-----------#
    confidence = np.array(result.boxes.conf.cpu())
    print("this is confidence:",confidence)

    # --------- anarray of bounding boxes---------------#
    bboxes = np.array(result.boxes.xyxy.cpu(),dtype="int")
    print("this is boxes",bboxes)

    # -------- getting indexes of the detections containing persons--------#
    idx = []
    for i in range(0,len(classes)):
        if classes[i] == 0:
            idx.append(i)

    print("these are indexes:",idx)

    # ----------- bounding boxes for person detections---------------#
    bbox = [] 
    for i in idx:
        temp = bboxes[i]
        print ("this is temp",temp)
        bbox.append(temp)
      
    # Convert to bbox to multidimensional list
    box_multi_list = [arr.tolist() for arr in bbox]
    print("this are final human detected boxes")
    print(box_multi_list)    

    # ------------ drawing of bounding boxes-------------#
    for box in box_multi_list :
        (x,y,x2,y2) = box
        
        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),2)
        cx = int((x+x2)/2)
        cy = int((y+y2)/2)
        centr_pt_cur_fr.append((cx,cy))
        cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

    print("this are the centroids in the current frame")
    print(centr_pt_cur_fr)

    # ------------- counting of total people in the footage ------------# 
    head_count =len(centr_pt_cur_fr)

    info = [("TOTAL VISIBLE PEOPLE",head_count)]

    for (x,y) in info:
        text = "{}: {}".format(x, y)
        cv2.putText(frame, text, (50,30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()                    
