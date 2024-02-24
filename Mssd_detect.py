import cv2
import numpy as np

def process(video_path):
    protox = "MobileNetSSD_deploy.prototxt"
    model = "MobileNetSSD_deploy.caffemodel"
    min_confidence = 0.4
    classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

            
    cap = cv2.VideoCapture(video_path)
    net = cv2.dnn.readNetFromCaffe(protox, model)
    # centr_pts_pre_fr = []
    while (True):
        
        ret,frame = cap.read()
        if not ret:
            break
        
        # frame_no +=1
        # image = cv2.imread("tough.jpg")
        frame = cv2.resize(frame , (600,600))
        centr_pts_cur_fr = []
        (h,w) = frame.shape[:2]
        print("height:",h, "width:",w)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame , (400,400)), 0.007843,(300,300),127.5)
        net.setInput(blob)
        detect = net.forward()
        print("this are detections")
        print (detect)
        # print(np.array(detect.shape[2])) 
        # print(" shape of array is" , np.array(detect.shape)) 
        for i in np.arange(0,detect.shape[2]):
            confidence = detect[0,0,i,2]
            if confidence >= min_confidence:
                idx = detect[0,0,i,1]
                if classes[int(idx)] != "person":
                    continue
                # count +=1
            
                bboxes = detect[0,0,i,3:7] * np.array([w,h,w,h])
                (x1,y1,x2,y2) = bboxes.astype("int")
                cx = int((x1+ x2)/2)
                cy = int((y1 + y2)/2)
                centr_pts_cur_fr.append((cx,cy))
                print( "start x:",x1,"start y:",y1,"end x:",x2, "end y :",y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                # cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
               
                # -------------------- ORIGINAL PART ENDS-------------------
                # print ("the dictionary is:",tracking_objects)           
                head_count =len(centr_pts_cur_fr)
                print("head count is : ",str(head_count))
                # ---------------displaying of text------------------
                info = [("TOTAL VISIBLE PEOPLE",head_count)]

                for (x,y) in info:
                    text = "{}: {}".format(x, y)
                    cv2.putText(frame, text, (50,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)            

                # print ("tracking objects")
                # print(tracking_objects)
                # print("frame number:",frame_no)
                # print("current frame :")
                # print(centr_pts_cur_fr)

                # print("current frame new points :")
                # print(centr_pts_cur_fr)

                # print("previous frame:")
                # print(centr_pts_pre_fr)

                # centr_pts_pre_fr = centr_pts_cur_fr_copy.copy()

        cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
     
process("enter ip address of camera or path of video saved")