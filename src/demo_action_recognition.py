#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from openvino_models import *


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
    dnn_action_encoder = ActionRecognitionEncoder(models_dir)
    dnn_action_decoder = ActionRecognitionDecoder(models_dir)
    embeddings = [] # A stack of frame embeddings.
    action_classes = []
    with open("/home/pcms/catkin_ws/src/mr_dnn/src/action_classes.txt", "r") as f:
        action_classes = [x.strip() for x in f.readlines()]

    # MAIN LOOP
    rospy.sleep(1)
    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        frame = _frame.copy()
        canvas = _frame.copy()

        # OpenVINO
        embedding = dnn_action_encoder.forward(frame)
        embeddings.append(embedding)
        while len(embeddings) > 16: embeddings.pop(0)
        if len(embeddings) == 16:
            out = dnn_action_decoder.forward(embeddings)
            for i, index in enumerate(out, 1):
                cv2.putText(canvas, action_classes[index], (5, 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # show image
        cv2.imshow("frame", canvas)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
