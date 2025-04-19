#!/usr/bin/env python3
import dt_apriltags
import cv2
import tf
from cv_bridge import CvBridge

import os
import rospy
from duckietown.dtros import DTROS, NodeType
import numpy as np
from duckietown_msgs.msg import BoolStamped, VehicleCorners
from geometry_msgs.msg import Point32

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import String

# import required libraries
import rospy
from duckietown.dtros import DTROS, NodeType
from navigate_template import NavigationControl

class ObstacleAvoidance(DTROS):
    def __init__(self, node_name):
        super(ObstacleAvoidance, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.detect_crosswalks = True
        self.drive_dist = 0
        
        self.nav = NavigationControl()

        # this is a hacky way to do timing, would be better to use rospy.time()....
        # but this works well enough with less effort/vars
        self.stop_time = 0

        # subscribe to camera feed
        self.img_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.img_sub = rospy.Subscriber(self.img_topic, CompressedImage, self.camera_callback, queue_size = 1)

        # State tracking
        self.tof_range = float('inf')
        self.maneuvering = False
        self.maneuver_state = 0
        self.state_time = 0

        # self.detect_crosswalks = True
        # self.crosswalk_stop_time = 0
        # self.crosswalk_drive_time = 0

        # Subscribers
        self.tof_sub = rospy.Subscriber(f"/{self._vehicle_name}/tof_driver_node/range",
                                        Range, self.tof_callback, queue_size=1)
        

    def detect_line(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_ranges = {'lower': np.array([100, 150, 50]), 'upper': np.array([140, 255, 255])}
        

        mask = cv2.inRange(hsv, red_ranges['lower'], red_ranges['upper'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Estimate distance based on contour position
                image_height = image.shape[0]
                distance = (image_height - (y + h)) / image_height
                
                return True, distance * 100
                    
        return False, float('inf')

    def detect_ducks(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        duck_ranges = {'lower': np.array([9, 91, 163]), 'upper': np.array([22, 255, 255])}
    
        mask = cv2.inRange(hsv, duck_ranges['lower'], duck_ranges['upper'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.contourArea(largest_contour) > 500
                    

    def maneuver_around_bot(self):
        self.state_time += 1
        turn_angle = 10
        turn_time = 12
        straight_time = 50
        if self.state_time < 5:
            return 0, 0
        if self.maneuver_state == 0:
            if self.state_time > 25:
                self.maneuver_state += 1
                self.state_time = 0
            return 0, 0
        elif self.maneuver_state == 1:
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            return -0.25, turn_angle
        elif self.maneuver_state == 2:
            if self.state_time > straight_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, 0
        elif self.maneuver_state == 3:
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, -turn_angle
        elif self.maneuver_state == 4:
            if self.state_time > straight_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, 0
        elif self.maneuver_state == 5:
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, -turn_angle
        elif self.maneuver_state == 6:
            if self.state_time > straight_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, 0
        elif self.maneuver_state == 7:
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, turn_angle
        else:
            self.maneuvering = False
            self.state_time = 0
            self.maneuver_state = 0
            return 0.2, 0

    def tof_callback(self, msg):
            self.tof_range = msg.range

    def camera_callback(self, msg):
        data_arr = np.frombuffer(msg.data, np.uint8)
        col_img = cv2.imdecode(data_arr, cv2.IMREAD_COLOR)
        vel = 0.5
        omega = 0

        # Crosswalk logic
        if self.detect_crosswalks or self.stop_time < 50:
            stopwalk_detection, _ = self.detect_line(col_img)
            if stopwalk_detection:
                self.detect_crosswalks = False
                vel = 0
                self.stop_time += 1
        else:
            if self.detect_ducks(col_img):
                vel = 0
            elif self.drive_dist < 50:
                vel = 0.5
                self.drive_dist += 1
            else:
                self.detect_crosswalks = True
                self.drive_dist = 0
                self.stop_time = 0

        # Obstacle logic
        if self.tof_range < 0.3 or self.maneuvering:
            self.maneuvering = True
            vel, omega = self.maneuver_around_bot()

        self.nav.publish_velocity(vel, omega)


if __name__ == '__main__':
    node = ObstacleAvoidance(node_name='obstacle_avoidance_node')
    rospy.spin()