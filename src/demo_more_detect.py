#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from openvino_models import *
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
    torch_home = "/home/pcms/models/pytorch/"
    dnn_yolo = Yolov5(torch_home)

    # OpenVINO
    models_dir = "/home/pcms/models/"
    dnn_attrs = PersonAttributesRecognition(models_dir)

    # MAIN LOOP
    rospy.sleep(1)
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        frame = _frame.copy()
        canvas = _frame.copy()

        # OpenVINO
        boxes = dnn_yolo.forward(frame)
        for id, index, conf, x1, y1, x2, y2 in boxes:
            if dnn_yolo.labels[index] != "person": continue

            person = frame[y1:y2, x1:x2, :]
            attrs = dnn_attrs.forward(person)
            print(attrs)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # show image
        cv2.imshow("frame", canvas)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
