#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from PIL import Image
from clip_interrogator import Config, Interrogator
import os 


def callback_path(msg):
    global _queue_path
    _queue_path.append(msg.data)


if __name__ == "__main__":
    rospy.init_node("image2text")
    rospy.loginfo("image2text node start!")

    _queue_path = []
    topic_path = "/image2text/path"
    rospy.Subscriber(topic_path, String, callback_path)

    topic_result = "/image2text/result"
    pub_result = rospy.Publisher(topic_result, String, queue_size=10)

    cfg = Config(cache_path="/home/pcms/models/openai/")
    cfg.apply_low_vram_defaults()
    cfg.device = "cpu"

    rospy.loginfo("Loading model with device %s." % cfg.device)
    ci = Interrogator(cfg)
    rospy.loginfo("image2text node is ready!")

    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if len(_queue_path) == 0: continue

        rospy.loginfo("%d in queue. %s" % (len(_queue_path), _queue_path[0]))
        path = _queue_path.pop(0)
        if not os.path.exists(path):
            rospy.logwarn("path: %s is not exists." % path)
            continue
        image = Image.open(path).convert("RGB")
        result = ci.generate_caption(image)
        rospy.loginfo(result)
        pub_result.publish(result)

    rospy.loginfo("image2text node end!")
