import cv2
from ultralytics import YOLO
import numpy as np
import time
import RPi.GPIO as GPIO
import time

# tracker 객체 생성 multiple
def create_tracker(frame,rect):
    tracker_temp = cv2.legacy.TrackerMOSSE.create()
    tracker_temp.init(frame,rect)
    return tracker_temp

def init_GPIO_PWM():
    global GPIO
    
    #Use pin 12 for PWM signal
    pwm_gpio_0 = 12
    pwm_gpio_1 = 32
    
    frequence = 50
    GPIO.setup(pwm_gpio_0, GPIO.OUT)
    GPIO.setup(pwm_gpio_1, GPIO.OUT)
    
    pwm_0 = GPIO.PWM(pwm_gpio_0, frequence)
    pwm_1 = GPIO.PWM(pwm_gpio_1, frequence)

    return [pwm_0,pwm_1]

def stop_GPIO():
    #Close GPIO & cleanup
    # pwm_0.stop()
    # pwm_1.stop()
    GPIO.cleanup()

def door_sleep(sleep_time):
        for i in range(sleep_time):
            print(f'Door Waiting .... {i+1}')
            time.sleep(1)


#  if count == 100 create new detection 
def detection_occasionally():
    global tracker_list
    global yolo_model
    global ret
    global img
    global door_timeout
    
    detection_flag = 1
    detection_tolerance = 0
    detected_rects = []
    
    tracker_list_temp = []
    
    while detection_flag:
        
        ret,img = capture.read()
        time.sleep(0.1)

        if not ret:
            exit()

        if detection_tolerance > 3:
            detection_tolerance = 0
            detection_flag = 0
            break

            
        detection_results = yolo_model(img,imgsz=960,conf=0.6)[0]
        
        if len(detection_results)<=0:
            detection_tolerance = detection_tolerance + 1
            continue
        else:      
            for detection_result in detection_results:

                detected_class_num = int(detection_result.boxes.cls[0])
                detected_class_string = detected_class_num
                
                crop_xywh = detection_result.boxes.xywh
                
                x = int(crop_xywh[0][0])
                y = int(crop_xywh[0][1])
                w = int(crop_xywh[0][2])
                h = int(crop_xywh[0][3])
                
                tracker = create_tracker(img,(x,y,w,h))
                detected_rects.append([tracker,detected_class_string])
                
            for rect in detected_rects:
                coor = rect[0]
                label = rect[1]
                tracker_list_temp.append([coor,label])
                
            if (len(tracker_list) <= len(tracker_list_temp)):
                tracker_list = tracker_list_temp
                
            detection_flag = 0

# -------- code starts here --------#

# GPIO setting
GPIO.setmode(GPIO.BOARD) #Use Board numerotation mode
GPIO.setwarnings(False) #Disable warnings

init_result = init_GPIO_PWM()
pwm_0 = init_result[0]
pwm_1 = init_result[1]

# get custom trained model
tracker_list = []
yolo_model = YOLO('ELSA_SEG_model.pt')
yolo_class_dict = {0: 'cane', 1: 'crutches', 2: 'walker', 3: 'wheelchair', 4: 'white_cane'}
yolo_class_dict_reverse = {'cane':0, 'crutches':1, 'walker':2, 'wheelchair':3, 'white_cane':4}

# get video
video_path = 'Wheelchair_Elevator_inside_in.mp4'
#video_path = 'Elevator camera video sample.mp4'
capture = cv2.VideoCapture(video_path)

detection_timeout = 0
tracker_len_prev = 0
door_timeout = 0

is_door_closed = 1



pwm_0.start(0)
pwm_1.start(0)
time.sleep(1)

pwm_0.ChangeDutyCycle(6.7)
pwm_1.ChangeDutyCycle(7.8)
time.sleep(0.9)

pwm_0.start(0)
pwm_1.start(0)
time.sleep(1)

print("Door opened")

is_door_closed = 0
inf_result = 0
normal_person_count = 0
target_person_count = 0
tracker_instance  = 0
no_target_timout = 0


while True:
    if is_door_closed:
        break        
        
    ret, img = capture.read()
    
    if not ret:
        exit()

    #print(f'no target timeout: {no_target_timout}')

    # 0번쨰 그리고 매 100번 마다 
    if ((detection_timeout == 0) or (detection_timeout==100)):
        detection_timeout = 1
        detection_occasionally()
        
    else:
        
        if no_target_timout >= 110:
            no_target_timout = 0
            door_timeout = 999

        if len(tracker_list)<=0:
            no_target_timout = no_target_timout + 1
        

        for tracker in tracker_list:
            tracker_instance = tracker[0]
            inf_result = tracker[1]
            
            if inf_result == 0:
                normal_person_count = normal_person_count + 1
            else:
                target_person_count = target_person_count + 1

            
            success, box = tracker_instance.update(img)
            
            x, y, w, h = [int(v) for v in box]
            
            left_top_x = x-80
            left_top_y = y-250

            right_low_x = x+w-80
            right_low_y = y+h-200

            text_cor = (left_top_x-10,left_top_y-10)
            text_font = cv2.FONT_HERSHEY_SIMPLEX
            text_font_scale = 1
            text_color = (0,0,255)
            text_thickness = 3
            
            cv2.putText(img, 
                        yolo_class_dict[inf_result], 
                        text_cor, 
                        text_font,
                        text_font_scale,
                        text_color,
                        text_thickness
                       )
            
            cv2.rectangle(img, (left_top_x,left_top_y), (right_low_x,right_low_y), color=(0,255,0), thickness=2)

    #print(f'time_out : {door_timeout}, prev : {tracker_len_prev}, now: {len(tracker_list)}, inf_result: {len(tracker_list)}')
    
    if tracker_len_prev != len(tracker_list):
        tracker_len_prev = len(tracker_list)
        door_timeout = 0
        
    if tracker_len_prev == len(tracker_list):
        door_timeout = door_timeout + 1
        
    if door_timeout >= 150:
        if len(tracker_list)==0 or normal_person_count>0:
            normal_person_count = 0
            print('\n Not detected and close_door after 5 seconds')
            door_sleep(5)
            is_door_closed = 1
            pwm_0.start(0)
            pwm_1.start(0)
            time.sleep(1)
        
            pwm_0.ChangeDutyCycle(8)
            pwm_1.ChangeDutyCycle(6.7)
            time.sleep(0.8)
            print("Door closed")
            pwm_0.stop()
            pwm_1.stop()
        else:
            target_person_count = 0
            print('\n Detected and close_door after 15 seconds \n')
            door_sleep(15)
            pwm_0.start(0)
            pwm_1.start(0)
            time.sleep(1)
        
            pwm_0.ChangeDutyCycle(8)
            pwm_1.ChangeDutyCycle(6.7)
            time.sleep(0.8)
            pwm_0.stop()
            pwm_1.stop()
            print("Door closed")
            is_door_closed = 1

    detection_timeout = detection_timeout + 1
    
    cv2.imshow('img', img)    
    if cv2.waitKey(1) == ord('q'):
        break
        

capture.release()
cv2.destroyAllWindows()
stop_GPIO()