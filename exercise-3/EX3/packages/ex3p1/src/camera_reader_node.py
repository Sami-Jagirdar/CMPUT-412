#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
import numpy as np

import cv2
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self._distorted_img_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._undistorted_img_topic = f"/{self._vehicle_name}/camera_node/undistorted_image/compressed"
        
        # bridge between OpenCV and ROS (to convert ROS images to OpenCV format)
        self.bridge = CvBridge()
        
        # # create window
        # self._window = "ex3p1-camera-reader"
        # cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        
        # construct subscriber - subcribe to camera intrinsic parameters topic
        self.sub = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)

        # construct subscriber - subcribe to distorted images topic
        self.sub = rospy.Subscriber(self._distorted_img_topic, CompressedImage, self.image_callback)

        # construct publisher to publish undistorted image to custom topic
        self.pub = rospy.Publisher(self._undistorted_img_topic, Image, queue_size=10)

        # Camera calibration parameters (to be set after receiving CameraInfo)
        self.camera_matrix = None
        self.dist_coeffs = None

    # def callback(self, msg):
        
    #     # convert JPEG bytes to CV image
    #     image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
    #     # display frame
    #     cv2.imshow(self._window, image)
    #     cv2.waitKey(1)

    def camera_info_callback(self, msg):
        """ To retrieve camera intrinsic parameters from CameraInfo. """
        # Extract camera matrix (K) and distortion coefficients (D)
        self.camera_matrix = np.array(msg.K).reshape(3, 3)  # 3x3 matrix
        self.dist_coeffs = np.array(msg.D)  # Distortion coefficients

        rospy.loginfo("Camera calibration intrinsic parameters received.")

    def image_callback(self, msg):
        """ To receive distorted images, undistort them, and publish. """
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.logwarn("Camera parameters not yet received. Skipping frame.")
            return
        
    # Try except debug
    # try:
    
        # Convert ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Get image size
        height, width = cv_image.shape[:2]

        # Compute new optimal camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (width, height), 1, (width, height)
        )

        # Undistort the image
        undistorted_img = cv2.undistort(
            cv_image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )

        # Convert back to ROS Image message
        undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted_img, encoding="bgr8")

        # Publish the undistorted image
        self._undistorted_img_topic.publish(undistorted_msg)

        rospy.loginfo("Published undistorted image.")

    # except Exception as e:
    #     rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
