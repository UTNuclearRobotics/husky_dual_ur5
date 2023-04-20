import itertools
import numpy as np
from sensor_msgs.msg import CameraInfo
from pathlib import Path
import rospy
from trac_ik_python.trac_ik import IK

from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from vgn.detection import *
from vgn.perception import UniformTSDFVolume

from active_grasp.timer import Timer
from active_grasp.rviz import Visualizer

import ipdb

class visualizer:
    def __init__(self):
        self.load_parameters()
        self.init_visualizer()
    
    def load_parameters(self):
        self.base_frame = "base_link" #rospy.get_param("~base_frame_id")
        self.cam_frame = rospy.get_param("~camera/frame_id")
        self.task_frame = "task"
        info_topic = rospy.get_param("~camera/info_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        self.intrinsic = from_camera_info_msg(msg)

    def init_visualizer(self):
        self.vis = Visualizer()

    def activate(self, bbox):
        self.vis.clear()

        self.bbox = bbox

        self.calibrate_task_frame()
        self.vis.bbox(self.base_frame, self.bbox)    

    def calibrate_task_frame(self):
        xyz = np.r_[self.bbox.center[:2] - 0.15, self.bbox.min[2] - 0.05]
        self.T_base_task = Transform.from_translation(xyz)
        self.T_task_base = self.T_base_task.inv()
        tf.broadcast(self.T_base_task, self.base_frame, self.task_frame)
        rospy.sleep(1.0)  # Wait for tf tree to be updated
        self.vis.roi(self.task_frame, 0.3)



vis = Visualizer()

class AABBox:
    def __init__(self, bbox_min, bbox_max):
        self.min = np.asarray(bbox_min)
        self.max = np.asarray(bbox_max)
        self.center = 0.5 * (self.min + self.max)
        self.size = self.max - self.min

    @property
    def corners(self):
        return list(itertools.product(*np.vstack((self.min, self.max)).T))

    def is_inside(self, p):
        return np.all(p > self.min) and np.all(p < self.max)

bbox = AABBox([5,0,0], [7,2,2])
vis.activate(bbox)