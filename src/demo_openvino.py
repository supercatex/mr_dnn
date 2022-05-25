#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from openvino_models import *
from rospkg import RosPack


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
    dnn_age_gender = AgeGenderRecognition(models_dir)
    dnn_emotions = EmotionsRecognition(models_dir)
    dnn_human_pose = HumanPoseEstimation(models_dir)
    dnn_face_reid = FaceReidentification(models_dir)
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")

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
            age, gender = dnn_age_gender.forward(face)
            emotion = dnn_emotions.forward(face)

            gender = dnn_age_gender.genders_label[gender]
            emotion = dnn_emotions.emotions_label[emotion]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, "Age: %d" % age, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(canvas, gender, (x1 + 5, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(canvas, emotion, (x1 + 5, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
            face_id = dnn_face_reid.forward(face)
            dist = dnn_face_reid.compare(Kinda, face_id)
            cv2.putText(canvas, "Kinda" if dist < 0.3 else "Unknown", (x1 + 5, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            

        poses = dnn_human_pose.forward(frame)
        canvas = dnn_human_pose.draw_poses(canvas, poses, 0.1)
        for pose in poses:
            for i, p in enumerate(pose):
                x, y, c = map(int, p)
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

        # show image
        cv2.imshow("frame", canvas)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
