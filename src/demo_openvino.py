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

    _DEVICE_NAME = rospy.get_param("~device")
    _DNN_FACE = rospy.get_param("~dnn_face")
    _DNN_FACE_ATTRS = rospy.get_param("~dnn_face_attrs")
    _DNN_POSE = rospy.get_param("~dnn_pose")
    _DNN_YOLO = rospy.get_param("~dnn_yolo")
    _DEVICE_NAME = "CPU"
    rospy.loginfo("Using %s device." % _DEVICE_NAME)

    # ROS Topics
    _image = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image)
    rospy.loginfo("waiting for topic: /camera/color/image_raw")
    rospy.wait_for_message("/camera/color/image_raw", Image)

    # OpenVINO
    rospy.loginfo("Loading models...")
    dnn_face = FaceDetection(device_name=_DEVICE_NAME)
    dnn_age_gender = AgeGenderRecognition(device_name=_DEVICE_NAME)
    dnn_emotions = EmotionsRecognition(device_name=_DEVICE_NAME)
    dnn_human_pose = HumanPoseEstimation(device_name=_DEVICE_NAME)
    dnn_face_reid = FaceReidentification(device_name=_DEVICE_NAME)
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo = Yolov8("yolov8n-seg", device_name=_DEVICE_NAME)

    # MAIN LOOP
    rospy.sleep(1)
    fps, fps_n = 0, 0
    while not rospy.is_shutdown():
        t1 = time.time()
        rospy.Rate(20).sleep()
        image = _image.copy()
        frame = _image.copy()
        
        # OpenVINO
        # Face
        if _DNN_FACE:
            boxes = dnn_face.forward(image)
            for x1, y1, x2, y2 in boxes:
                if _DNN_FACE_ATTRS:
                    face = image[y1:y2, x1:x2, :].copy()
                    age, gender = dnn_age_gender.forward(face)
                    emotion = dnn_emotions.forward(face)

                    gender = dnn_age_gender.genders_label[gender]
                    emotion = dnn_emotions.emotions_label[emotion]
                    cv2.putText(frame, "Age: %d" % age, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, gender, (x1 + 5, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, emotion, (x1 + 5, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    face_id = dnn_face_reid.forward(face)
                    dist = dnn_face_reid.compare(Kinda, face_id)
                    cv2.putText(frame, "Kinda" if dist < 0.3 else "Unknown", (x1 + 5, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Pose
        if _DNN_POSE:
            poses = dnn_human_pose.forward(image)
            frame = dnn_human_pose.draw_poses(frame, poses, 0.1)
            for pose in poses:
                for i, p in enumerate(pose):
                    x, y, c = map(int, p)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Yolov8
        if _DNN_YOLO:
            detections = dnn_yolo.forward(image)
            # print(detections)
            for i, detection in enumerate(detections):
                frame = draw_results(detection, frame, dnn_yolo.classes)
                dnn_yolo.draw_bounding_box(detection, frame)
        
        t2 = time.time()
        fps = (fps * fps_n + 1.0 / (t2 - t1)) / (fps_n + 1)
        fps_n += 1
        cv2.putText(frame, "%.2ffps" % (fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)        
        
        # show image
        h, w = frame.shape[:2]
        frame2 = cv2.resize(frame, (w * 2, h * 2))
        cv2.imshow("frame", frame2)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
