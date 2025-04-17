#!/usr/bin/env python3

import rospy
import os

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = True
ENGLISH = False
SAFETY = False
AUSSIE = False


class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']
        # self.veh = rospy.get_param("~veh")

        # Publishers & Subscribers
        if SAFETY:
            self.tof_sub = rospy.Subscriber("/" + self.veh + "/front_center_tof_driver_node/range",
                                            Range,
                                            self.cb_tof,
                                            queue_size=1)
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -180
        else:
            self.offset = 220
        if AUSSIE:
            self.offset = 0
        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.028
        self.D = -0.0025
        self.I = 0

        if AUSSIE:
            self.P = 0.0005
            self.D = -0.025
            self.I = 0.5

        self.last_error = 0
        self.integral = 0
        self.last_time = rospy.get_time()
        self.tof_distance = 1.0
        self.obj_stop = False
        # Wait a little while before sending motor commands
        rospy.Rate(0.20).sleep()

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def cb_tof(self,  msg):
        self.tof_distance = msg.range
        if 0.05 < self.tof_distance <= 0.3:
            self.obj_stop = True

    def callback(self, msg):
        img = self.jpeg.decode(msg.data)
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Try detecting yellow lane first
        yellow_mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        # yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Fallback to white lane if yellow not detected
        def find_largest_contour(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_area = 20
            max_idx = -1
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > max_area:
                    max_idx = i
                    max_area = area
            return contours, max_idx

        # Search yellow lane first
        contours, max_idx = find_largest_contour(yellow_mask)
        following_white = False

        if max_idx != -1:
            mask_to_use = yellow_mask
        else:
            # Try white lane
            white_lower = np.array([120, 18, 155], np.uint8)
            white_upper = np.array([128, 39, 255], np.uint8)
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            contours, max_idx = find_largest_contour(white_mask)
            mask_to_use = white_mask
            following_white = max_idx != -1

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # If following white lane, invert the offset
                offset = -(self.offset+50) if following_white else self.offset
                self.proportional = cx - int(crop_width / 2) + offset

                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (255, 0, 0) if following_white else (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                self.proportional = None
        else:
            self.proportional = None

        if DEBUG:
            debug_masked = cv2.bitwise_and(crop, crop, mask=mask_to_use)
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_masked))
            self.pub.publish(rect_img_msg)


    def drive(self):
        if self.obj_stop:
            self.stop(8)
            self.obj_stop = False
            self.logwarn("About to stop")
            rospy.sleep(1.0)
            self.logwarn("Stopped for two second")

        elif self.proportional is None:
            self.twist.omega = 0
            self.last_error = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            # I term
            current_time = rospy.get_time()
            dt = current_time - self.last_time
            if dt <= 0:
                I = 0
            else:
                self.integral += self.proportional * dt
                I = self.I * self.integral
            

            self.twist.v = self.velocity
            self.twist.omega = P + D + I
            # if DEBUG:
            #     self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

        self.vel_pub.publish(self.twist)

    def stop(self, duration):
        self.twist.v = 0
        self.twist.omega = 0
        for i in range(duration):
            self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()