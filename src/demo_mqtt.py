#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *
from rospkg import RosPack
from paho.mqtt.client import Client
from netifaces import interfaces, ifaddresses, AF_INET


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

    # MQTT
    ip = "127.0.0.1"
    for name in interfaces():
        addresses = [i['addr'] for i in ifaddresses(name).setdefault(AF_INET, [{'addr': '127.0.0.1'}] )]
        if name[:2] == "wl": ip = addresses[0]
    print(ip)

    client = Client()
    client.connect(ip, 1883, 60)

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
        
        # MQTT
        client.publish("cmd", "1" if len(boxes) > 0 else "0")

        # show image
        cv2.imshow("frame", canvas)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
