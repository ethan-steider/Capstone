import cv2
import os
import time

# TODO implement NAME, so we can get which camera has errors
class SAURON_CAM:
    def __init__(self, ID, SAVEPATH=""):
        self.ID = ID
        self.SAVEPATH = os.path.join(os.getcwd(),SAVEPATH)
        self.device = cv2.VideoCapture(ID)

    def capture_frame(self):
        self.ret, self.frame = self.device.read()
        if not (self.ret):
            raise Exception(f"CAMERA {getID()} FAILED TO GRAB FRAME")
            
        return self.frame

    def getID(self):
        return self.ID

    def setID(self, ID):
        self.ID = ID

    def getSAVEPATH(self):
        return self.SAVEPATH

    def setSAVEPATH(self, SAVEPATH):
        self.SAVEPATH = SAVEPATH

    def getFrame(self):
        return self.frame

    def setFrame(self, frame):
        self.frame = frame

    def release(self):
        self.device.release()

if __name__ == "__main__":
    WEBCAM = SAURON_CAM(0)
    cv2.namedWindow("test_1")
    img_counter = 0
    
    while True:
        
        try:
            WEBCAM.capture_frame()
        except Exception as error:
            print('Caught this error: ' + repr(error))
            break
        frame_0 = WEBCAM.getFrame()
        cv2.imshow("test_0",frame_0)
        k = cv2.waitKey(1)
        if k%256==27:
            # ESC pressed
            print("ESC pressed, closing...")
            break
        elif k%256==32:
            # SPACE pressed
            img_name_0 = "WEBCAM{}.png".format(img_counter)
            cv2.imwrite(img_name_0,frame_0)
            print("{} written".format(img_name_0))
            img_counter += 1
    WEBCAM.release()
    cv2.destroyAllWindows()
