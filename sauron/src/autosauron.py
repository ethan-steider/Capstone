import cv2
import os
import time 
import camapi
from datetime import datetime

LWIR_ID = 1
LWIR_SAVE_PATH = "lwir_frames" #add path to where we want to store the LWIR photos
LWIR_NAME = "LWIR"


RGB_ID = 0
RGB_SAVE_PATH = "rgb_frames" #add path to where we want to store the LWIR photos
RGB_NAME = "RGB"

lwir_cam = camapi.SAURON_CAM(LWIR_ID, LWIR_SAVE_PATH)
rgb_cam = camapi.SAURON_CAM(RGB_ID, RGB_SAVE_PATH)


while True:
    
    lwir_frame = lwir_cam.capture_frame()
    DATE = datetime.now().strftime('%Y%m%d_%H-%M-%S-%f')
    #t_image1 = time.time() 
    rgb_frame = rgb_cam.capture_frame()
    #t_image2 = time.time() 

    print(DATE + " Saving Frames")
    cv2.imwrite(os.path.join(lwir_cam.getSAVEPATH(),DATE + "_"+LWIR_NAME+".png"), lwir_frame)
    cv2.imwrite(os.path.join(rgb_cam.getSAVEPATH(),DATE + "_"+RGB_NAME+".png"), rgb_frame)

    #t_save = time.time() 
    #print(t_image1, t_image2, t_save)

    #print("waiting")
    #time.sleep(1)
