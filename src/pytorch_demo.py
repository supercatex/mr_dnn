#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pytorch_models import *


def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    # ROS Topics
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    _frame = rospy.wait_for_message("/camera/rgb/image_raw", Image)

    # PyTorch
    torch_home = "/home/pcms/models/torch/"
    dnn_objs = FasterRCNN(torch_home)
    dnn_poses = KeypointRCNN(torch_home)

    # MAIN LOOP
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        frame = _frame.copy()

        # Torch
        # boxes = dnn_objs.forward(frame)
        # for id, label, conf, x1, y1, x2, y2 in boxes:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, str(label), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        poses = dnn_poses.forward(frame)

        # show image
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
