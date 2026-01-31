import cv2
cam_0 = cv2.VideoCapture(0)
cam_1 = cv2.VideoCapture(1)
cv2.namedWindow("test_1")
cv2.namedWindow("test_0")
img_counter = 0

while True:
    ret_0,frame_0 = cam_0.read()
    ret_1,frame_1 = cam_1.read()
    if not (ret_0 or ret_1):
        print("failed to grab frame")
        break
    cv2.imshow("test_0",frame_0)
    cv2.imshow("test_1",frame_1)
    k = cv2.waitKey(1)
    if k%256==27:
        # ESC pressed
        print("ESC pressed, closing...")
        break
    elif k%256==32:
        # SPACE pressed
        img_name_0 = "rgb_frame_{}.png".format(img_counter)
        img_name_1 = "lwir_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name_0,frame_0)
        print("{} written".format(img_name_0))
        cv2.imwrite(img_name_1,frame_1)
        print("{} written".format(img_name_1))
        img_counter += 1
cam.release()
cv2.destroyAllWindows()
