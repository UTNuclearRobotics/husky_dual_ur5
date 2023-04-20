import itertools
import numpy as np
from sensor_msgs.msg import CameraInfo
from pathlib import Path
import rospy
from trac_ik_python.trac_ik import IK

from robot_helpers.ros import tf as tf_custom
from robot_helpers.ros.conversions import *
from vgn.detection import *
from vgn.perception import UniformTSDFVolume

from active_grasp.timer import Timer
from active_grasp.rviz import Visualizer
from active_grasp.policy import compute_error, solve_ik
from active_grasp.nbv import NextBestView, raycast
# from active_grasp.controller import ViewHalfSphere

import ipdb

from vgn.utils import look_at, cartesian_to_spherical, spherical_to_cartesian
import tf
import copy
import cv_bridge
from sensor_msgs.msg import Image

class visualizerVaultbot:
    def __init__(self):
        self.load_parameters()
        self.init_visualizer()
    
    def load_parameters(self):
        self.base_frame = "map" #"base_link" #rospy.get_param("~base_frame_id")
        self.cam_frame = "camera_link" #rospy.get_param("~camera/frame_id")
        self.task_frame = "map_offset" #"task"
        info_topic = "/my_temoto/robot_manager/robots/vaultbot_sim/camera/depth/camera_info" #rospy.get_param("~camera/info_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo, rospy.Duration(2.0))
        msg.header.frame_id = "camera_link"
        self.intrinsic = from_camera_info_msg(msg)
        

    def init_visualizer(self):
        self.vis = Visualizer(base_frame="map")
        self.tsdf = UniformTSDFVolume(5, 250)
        self.views = []

    def activate(self, bbox):
        self.vis.clear()

        self.bbox = bbox

        # self.calibrate_task_frame()
        # self.vis.bbox(self.base_frame, self.bbox)    
        

    def integrate(self, img, x, q=None):
        self.views.append(x)
        # self.vis.path(self.base_frame, self.intrinsic, self.views)
        
        # with Timer("tsdf_integration"):
            # self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)
            # ipdb.set_trace()
        # print
        self.tsdf.integrate(img, self.intrinsic, x)

        scene_cloud = self.tsdf.get_scene_cloud()
        self.vis.scene_cloud(self.task_frame, np.asarray(scene_cloud.points))

        map_cloud = self.tsdf.get_map_cloud()
        self.vis.map_cloud(
            self.task_frame,
            np.asarray(map_cloud.points),
            np.expand_dims(np.asarray(map_cloud.colors)[:, 0], 1),
        )
        # tsdf_grid = self.tsdf.get_grid()
        # out = self.vgn.predict(tsdf_grid)
        # self.vis.quality(self.task_frame, self.tsdf.voxel_size, out.qual, 0.9)


    def calibrate_task_frame(self):
        xyz = np.r_[self.bbox.center[:2] - 0.15, self.bbox.min[2] - 0.05]
        self.T_base_task = Transform.from_translation(xyz)
        self.T_task_base = self.T_base_task.inv()
        tf_custom.broadcast(self.T_base_task, self.base_frame, self.task_frame)
        rospy.sleep(1.0)  # Wait for tf tree to be updated
        self.vis.roi(self.task_frame, 5)


class wrap_rotation:
    def __init__(self, mat):
        self.mat = mat
    def as_matrix(self):
        return self.mat
    def translation(self):
        return (self.mat[:-1,-1])


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


class camera_mod:
    def __init__(self):
        self.depth_topic = "/my_temoto/robot_manager/robots/vaultbot_sim/camera/depth/image_raw"
        self.init_camera_stream()

    def init_camera_stream(self):
        self.cv_bridge = cv_bridge.CvBridge()
        rospy.Subscriber(self.depth_topic, Image, self.sensor_cb, queue_size=1)
        # ipdb.set_trace()

    def sensor_cb(self, msg):
        self.latest_depth_msg = msg
        self.img_latest = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        

class ViewHalfSphere:
    def __init__(self, bbox, min_z_dist):
        self.center = bbox.center
        # self.r = 0.5 * bbox.size[2] + min_z_dist
        self.r = 0.6 * bbox.size[2] + min_z_dist

    def get_view(self, theta, phi):
        eye = self.center + spherical_to_cartesian(self.r, theta, phi)
        up = np.r_[1.0, 0.0, 0.0]
        return look_at(eye, self.center, up)

    def sample_view(self):
        raise NotImplementedError
    

def main():
    rospy.init_node("visualizer")
    vis = visualizerVaultbot()
    tf_listener = tf.TransformListener()
    
    # bbox = AABBox([0.65,-0.3,0], [1.0,0.3,0.75])
    bbox = AABBox([0.5 ,-0.1,0], [1.2 ,0.1,0.5])
    rate = rospy.Rate(0.25)
    cam = camera_mod()


    xyz_base_link = np.array([-0.05, 0, -0.01])
    #Transform from map frame to base link frame
    b_link_map = Transform.from_translation(xyz_base_link)
    # tf_custom.broadcast(b_link_map, "base_link", "bbox_center")

    min_z_dist = rospy.get_param("~camera/min_z_dist")
    # generate views with respect to "map" frame
    view_sphere = ViewHalfSphere(bbox, min_z_dist)
    # pose_transform = np.hstack((np.vstack((np.eye(3),np.zeros((1,3)))), np.zeros((4,1))))
    # pose_transform[-1,-1] = 1

    #wait for first image message to subscirber so that camera_mod latest_depth_msg is not empty
    rospy.wait_for_message("/my_temoto/robot_manager/robots/vaultbot_sim/camera/depth/image_raw", Image, rospy.Duration(1.0))
    # ipdb.set_trace()
    # vis.activate(bbox)

    policy = NextBestView()

    policy.activate(bbox, view_sphere)
    views = policy.generate_views(5)

    # find reachable views
    q_i = [-1.9656962874049415, -0.15482256751911017, -1.4108062006869355, -1.7107767066943467, 0.3629895048846805, 0.0686822943713814]
    views_reachable = []
    for view in views:
        view_b_link = b_link_map*view

        q_solve = policy.solve_cam_ik(q0=q_i, view=view_b_link)
        # e_solve = policy.solve_ee_ik(q0=q_i, pose=view_b_link)
        # print(f"q_solution = {q_solve} \n e_solution = {e_solve}")
        if not q_solve:
            continue
        else:
            # print(f"view translation component {view.translation}")
            views_reachable.append(view)

    views = views_reachable

    tf_listener.waitForTransform("/camera_link_dummy", "/map_offset", rospy.Time(0), rospy.Duration(1.5))
    (trans,rot) = tf_listener.lookupTransform("/camera_link_dummy", "/map_offset", rospy.Time(0))

    rot_matrix = tf.transformations.quaternion_matrix(rot)
    rot_matrix[:-1, -1] = trans

    #reconstruct initial scene
    policy.views.append(views)
    policy.tsdf.integrate(cam.img_latest, policy.intrinsic, wrap_rotation(rot_matrix))
    scene_cloud = policy.tsdf.get_scene_cloud()
    policy.vis.scene_cloud(policy.task_frame, np.asarray(scene_cloud.points))

    map_cloud = policy.tsdf.get_map_cloud()
    policy.vis.map_cloud(
        policy.task_frame,
        np.asarray(map_cloud.points),
        np.expand_dims(np.asarray(map_cloud.colors)[:, 0], 1),
    )
    ############################

    # ipdb.set_trace()
    policy.vis.ig_views(policy.base_frame, policy.intrinsic, views, np.random.rand(len(views)))
    while not rospy.is_shutdown():
        # print(bbox)
        # if (tf_listener.frameExists("/camera_link") and tf_listener.frameExists("/map")):



        # ipdb.set_trace()
        gains = [policy.ig_fn(v, policy.downsample) for v in views]
        costs = [policy.cost_fn(v) for v in views]
        utilities = gains / np.sum(gains) - costs / np.sum(costs)
        print(f"gains={gains}, costs={costs}")
        print(utilities)
        if not np.isnan(utilities).any():
            policy.vis.ig_views(policy.base_frame, policy.intrinsic, views, utilities)
        else:
            policy.vis.ig_views(policy.base_frame, policy.intrinsic, views, np.random.rand(len(views)))

        tf_listener.waitForTransform("/camera_link_dummy", "/map_offset", rospy.Time(0), rospy.Duration(1.5))
        (trans,rot) = tf_listener.lookupTransform("/camera_link_dummy", "/map_offset", rospy.Time(0))

        rot_matrix = tf.transformations.quaternion_matrix(rot)
        rot_matrix[:-1, -1] = trans
        # vis.integrate(cam.img_latest, wrap_rotation(rot_matrix), None)
        policy.views.append(views)
        policy.tsdf.integrate(cam.img_latest, policy.intrinsic, wrap_rotation(rot_matrix))

        scene_cloud = policy.tsdf.get_scene_cloud()
        policy.vis.scene_cloud(policy.task_frame, np.asarray(scene_cloud.points))

        map_cloud = policy.tsdf.get_map_cloud()
        policy.vis.map_cloud(
            policy.task_frame,
            np.asarray(map_cloud.points),
            np.expand_dims(np.asarray(map_cloud.colors)[:, 0], 1),
        )



        rate.sleep()


if __name__ == '__main__':
    main()