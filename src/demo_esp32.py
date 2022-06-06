#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *
from rospkg import RosPack
import requests


def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    # ROS Topics
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    _frame = rospy.wait_for_message("/camera/rgb/image_raw", Image)

    # OpenVINO
    models_dir = "/home/pcms/models/openvino/"
    dnn_face = FaceDetection(models_dir)

    # MAIN LOOP
    rospy.sleep(1)
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        frame = _frame.copy()
        canvas = _frame.copy()

        # OpenVINO
        boxes = dnn_face.forward(frame)
        for x1, y1, x2, y2 in boxes:
            face = frame[y1:y2, x1:x2, :].copy()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        requests.get("http://192.168.50.61/?led=%s" % ("on" if len(boxes) > 0 else "off"))

        # show image
        cv2.imshow("frame", canvas)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
