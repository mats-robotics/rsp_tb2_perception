#!/usr/bin/env python3

import rospy
import pyrealsense2
from detection_msgs.msg import BoundingBox, BoundingBoxes
from message_filters import (
    ApproximateTimeSynchronizer,
    Subscriber,
    TimeSynchronizer,
    Cache,
)
from obj_in_map_msgs.msg import ObjectInMap, ObjectsInMap
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from nav_msgs.msg import MapMetaData
from rostopic import get_topic_type
from cv_bridge import CvBridge
import time
import cv2
import numpy as np


class Detection2Dto3D:
    def __init__(self):
        self.objects = dict()
        self.bridge = CvBridge()
        # print(9)
        # subscribe to map metadata topic
        self.map_metadata = rospy.wait_for_message(
            rospy.get_param("~map_metadata_topic"), MapMetaData
        )
        self.process_map_metadata()
        # print(10)
        # subscribe to camera info topic
        self.camera_info = rospy.wait_for_message(
            rospy.get_param("~camera_info_topic"), CameraInfo
        )
        self.camera_intrinsics = pyrealsense2.intrinsics()
        self.process_camera_info()
        # print(11)
        # subscribe to get robot pose in world
        self.pose_sub = rospy.Subscriber(rospy.get_param("~robot_pose_topic"), PoseWithCovarianceStamped,self.robot_pose_callback)
        
        # subscriber to detection messages
        self.detection_sub = Subscriber(
            rospy.get_param("~detection_topic"), BoundingBoxes
        )
        # print(7)
        # subscribe to depth image topic
        depth_image_type, depth_image_topic, _ = get_topic_type(
            rospy.get_param("~depth_image_topic"), blocking=True
        )
        self.depth_compressed = depth_image_type == "sensor_msgs/CompressedImage"
        if self.depth_compressed:
            self.depth_sub = Subscriber(depth_image_topic, CompressedImage)
            raise ValueError(
                "Cannot use compressed depth topic with Python for this version"
            )
        else:
            self.depth_sub = Subscriber(depth_image_topic, Image)

        # print(3)

        # Filter message
        self.ts = TimeSynchronizer([self.depth_sub, self.detection_sub], queue_size=30)
        self.ts.registerCallback(self.ts_callback)
        
        # publisher to publish pose of each object
        self.objects_pub = rospy.Publisher(rospy.get_param("~objects_in_map_topic"), ObjectsInMap, queue_size=10)

        self.detections = None
        self.depth = None
        self.color = None

        time.sleep(3)
        print(5)


    def process_camera_info(self):
        # print(self.camera_info)
        self.camera_intrinsics = pyrealsense2.intrinsics()
        self.camera_intrinsics.width = self.camera_info.width
        self.camera_intrinsics.height = self.camera_info.height
        self.camera_intrinsics.ppx = self.camera_info.K[2]
        self.camera_intrinsics.ppy = self.camera_info.K[5]
        self.camera_intrinsics.fx = self.camera_info.K[0]
        self.camera_intrinsics.fy = self.camera_info.K[4]
        self.camera_intrinsics.model = pyrealsense2.distortion.none
        self.camera_intrinsics.coeffs = [i for i in self.camera_info.D]

    def process_map_metadata(self):
        self.map_resolution = self.map_metadata.resolution
        self.x_origin = self.map_metadata.origin.position.x
        self.y_origin = self.map_metadata.origin.position.y

    def convert_depth_to_phys_coord_using_realsense(self, x, y):

        depth = self.depth[y][x]
        # print(depth)
        result = pyrealsense2.rs2_deproject_pixel_to_point(
            self.camera_intrinsics, [x, y], depth
        )

        # convert result to robot convention
        return [result[2], -result[0], -result[1]]

    def robot_pose_callback(self, pose_msg):
        # preprocess robot position
        self.robot_x = pose_msg.pose.pose.position.x
        self.robot_y = pose_msg.pose.pose.position.y
        self.robot_yaw = np.arctan2(pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w)*2


    def ts_callback(self, depth_msg, detection_msg):
        print(1)
        if rospy.get_param("memory_active", True) == False:
            self.publish_objects()
            return

        # preprocess message
        self.depth = self.bridge.imgmsg_to_cv2(depth_msg)
        self.detections = detection_msg

        # preprocess robot position
        # robot_x = pose_msg.pose.pose.position.x
        # robot_y = pose_msg.pose.pose.position.y
        # robot_yaw = np.arctan2(pose_msg.pose.pose.orientation.w, pose_msg.pose.pose.orientation.z)*2

        # return if no detection
        if self.detections.bounding_boxes is None:
            return

        #
        for i in range(len(self.detections.bounding_boxes)):
            bounding_box = self.detections.bounding_boxes[i]
            object_name = bounding_box.Class

            obj_x, obj_y, obj_z = self.calculate_position_relative_to_camera(
                bounding_box.xmin,
                bounding_box.ymin,
                bounding_box.xmax,
                bounding_box.ymax,
            )
            ##### calculate wrt map #####
            # offset between camera at OM home pose and robot base
            obj_x = obj_x - 0.12
            
            # obj pose in map unit relative to robot frame
            obj_x_map_unit = obj_x #*self.map_resolution 
            obj_y_map_unit = obj_y #*self.map_resolution 
            # print(object_name, obj_x_map_unit, obj_y_map_unit)
            # print(obj_x, obj_y)
            # obj pose in map unit relative to map frame
            obj_x_map_unit_wrt_map = np.cos(self.robot_yaw)*obj_x_map_unit - np.sin(self.robot_yaw)*obj_y_map_unit + self.robot_x
            obj_y_map_unit_wrt_map = np.sin(self.robot_yaw)*obj_x_map_unit + np.cos(self.robot_yaw)*obj_y_map_unit + self.robot_y 

            # update dict
            self.objects[object_name] = (obj_x_map_unit_wrt_map, obj_y_map_unit_wrt_map)
            
        
        # publish message from dict
        self.publish_objects()

    def publish_objects(self):
        # print(1)
        # print(self.objects)
        # print(rospy.Time.now())
        objects_msg = ObjectsInMap()
        objects_msg.header.stamp = rospy.Time.now() 
        
        for key in self.objects:
            object_msg = ObjectInMap()
            object_msg.object_name = key 
            object_msg.x_in_map = self.objects[key][0]
            object_msg.y_in_map = self.objects[key][1]
            objects_msg.objects_in_map.append(object_msg)
        
        self.objects_pub.publish(objects_msg)

    def calculate_position_relative_to_camera(self, xmin, ymin, xmax, ymax):
        # using median of 9 points at the middle
        # return position in meters
        xmid = int((xmax + xmin) / 2)
        ymid = int((ymax + ymin) / 2)
        phys_coords = []
        for i in range(xmid - 1, xmid + 2):
            for j in range(ymid - 1, ymid + 2):
                phys_coord = self.convert_depth_to_phys_coord_using_realsense(i, j)
                phys_coords.append(phys_coord)
        phys_coords = np.array(phys_coords)
        # print(phys_coords)
        median_coords = np.median(phys_coords, axis=0) /1000
        # print(median_coords)
        return median_coords[0], median_coords[1], median_coords[2]


if __name__ == "__main__":
    rospy.init_node("detection_depth_to_pose", anonymous=True)
    detection_depth_to_pose = Detection2Dto3D()

    # print(detection_depth_to_pose.convert_depth_to_phys_coord_using_realsense(813, 544))

    rospy.spin()
