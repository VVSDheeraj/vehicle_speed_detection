
"""
In this implementation
we will demonstrate the assumption we have considered in the optical flow script using two lines (red and blue)
The distance assumed between these two lines is taken as 10 meters
This script will also count the number of cars moving up and down and will denote that number on the counter in the video.
This script can also be used to detect the speed of the cars, but since due to time constraints we were not able to finish 
using the 'bus', 'truck', 'bikes', 'trailer' classes from the COCO dataset we have not implemented the calculation 
of speed using this script.
The speed can be calculated by using the simple distance time formula
Distance = Speed / Time
"""

import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolo26s.pt')

cap = cv2.VideoCapture('highway.mp4')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

count = 0
tracker = Tracker()
down = {}
up = {}
counter_down = []
counter_up = []

# Creating a folder to save frames
if not os.path.exists('detected_frames_yolo'):
    os.makedirs('detected_frames_yolo')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_yolo.avi', fourcc, 20.0, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list = []
    red_line_y = 198
    blue_line_y = 268
    offset = 6
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        if red_line_y<(cy+offset) and red_line_y > (cy-offset):
           down[id]= time.time()   # current time when vehicle touch the first line
        if id in down:

           if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
             elapsed_time= time.time() - down[id]  
             if counter_down.count(id)==0:
                counter_down.append(id)
                distance = 70 
                a_speed_ms = distance / elapsed_time
                a_speed_kh = a_speed_ms * 3.6  
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        #####going UP#####
        if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
           up[id]=time.time()
        if id in up:
           if red_line_y<(cy+offset) and red_line_y > (cy-offset):
             elapsed1_time=time.time() - up[id]
             if counter_up.count(id)==0:
                counter_up.append(id)      
                distance1 = 70 
                a_speed_ms1 = distance1 / elapsed1_time
                a_speed_kh1 = a_speed_ms1 * 3.6
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)

    cv2.line(frame, (172, 198), (774, 198), red_color, 2)
    cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame, (50, 400), (974, 400), blue_color, 2)
    cv2.putText(frame, ('Blue Line'), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Saving the frames
    frame_filename = f'detected_frames_yolo/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    out.write(frame)

    # cv2.imshow("frames", frame)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break

cap.release()
out.release()
# cv2.destroyAllWindows()
print("Processing complete! Output saved to output_yolo.avi")
