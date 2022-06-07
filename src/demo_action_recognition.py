#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *
from pcms.model_labels import ACTION_CLASSES


def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    # ROS Topics
    _image = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    rospy.wait_for_message("/camera/rgb/image_raw", Image)

    # OpenVINO
    dnn_action_encoder = ActionRecognitionEncoder()
    dnn_action_decoder = ActionRecognitionDecoder()
    embeddings = [] # A stack of frame embeddings.
    action_classes = ACTION_CLASSES

    # MAIN LOOP
    rospy.sleep(1)
    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        image = _image.copy()
        frame = _image.copy()

        # OpenVINO
        embedding = dnn_action_encoder.forward(image)
        embeddings.append(embedding)
        while len(embeddings) > 16: embeddings.pop(0)
        if len(embeddings) == 16:
            out = dnn_action_decoder.forward(embeddings)
            for i, (index, value) in enumerate(out, 1):
                cv2.putText(frame, "%.2f %s" % (value, action_classes[index]), (5, 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # show image
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
