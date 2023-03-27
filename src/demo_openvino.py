#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *
from rospkg import RosPack
import time


def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    # ROS Topics
    _image = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    rospy.loginfo("waiting for topic: /camera/rgb/image_raw")
    rospy.wait_for_message("/camera/rgb/image_raw", Image)

    # OpenVINO
    dnn_face = FaceDetection()
    dnn_age_gender = AgeGenderRecognition()
    dnn_emotions = EmotionsRecognition()
    dnn_human_pose = HumanPoseEstimation()
    dnn_face_reid = FaceReidentification()
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo = Yolov8("yolov8s")

    # MAIN LOOP
    rospy.sleep(1)
    fps, fps_n = 0, 0
    while not rospy.is_shutdown():
        t1 = time.time()
        rospy.Rate(20).sleep()
        image = _image.copy()
        frame = _image.copy()

        # Yolov8
        detections = dnn_yolo.forward(frame)
        for i, detection in enumerate(detections):
            dnn_yolo.draw_bounding_box(detection, frame)
        
        t2 = time.time()
        fps = (fps * fps_n + 1.0 / (t2 - t1)) / (fps_n + 1)
        fps_n += 1
        cv2.putText(frame, "%.2ffps" % (fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)        
        
        # show image
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
