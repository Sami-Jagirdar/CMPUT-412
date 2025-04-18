#!/usr/bin/env python3

import rospy
import os
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point32
import cv2
from general_navigation import NavigationControl
from cv_bridge import CvBridge
from turbojpeg import TurboJPEG
import numpy as np

DEBUG_LANE_FOLLOW = True
DEBUG_TAIL = True

class TailDuckNode(DTROS):
    def __init__(self, node_name):
        super(TailDuckNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']
        self.process_frequency = 10

        # --- Duckiebot Tailing setup ---
        self.last_stamp = rospy.Time.now()
        self.circlepattern_dims = [7, 3]
        self.blobdetector_min_area = 10
        self.blobdetector_min_dist_between_blobs = 2
        self.cbParametersChanged()
        self.last_seen = rospy.Time(0)
        self.tail_timeout = rospy.Duration(0.6)
        self.tailing = False
        self.last_tail_error = (0,0)

        # --- Lane following setup ---
        self.ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
        self.offset = 220
        self.P = 0.028
        self.D = -0.0025
        self.I = 0
        self.last_error = 0
        self.integral = 0
        self.last_time = rospy.get_time()

        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.image_callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.pub_mask = rospy.Publisher(
            f"/{self.veh}/combined/mask/compressed", CompressedImage, queue_size=1
        )

        # self.pub_centers = rospy.Publisher("/{}/duckiebot_detection_node/centers".format(os.environ['VEHICLE_NAME']), VehicleCorners, queue_size=1)
        self.pub_circlepattern_image = rospy.Publisher("/{}/duckiebot_detection_node/detection_image/compressed".format(os.environ['VEHICLE_NAME']), CompressedImage, queue_size=1)
        # self.pub_detection = rospy.Publisher("/{}/duckiebot_detection_node/detection".format(os.environ['VEHICLE_NAME']), BoolStamped, queue_size=1)
        self.log("Detection Initialization completed.")

        self.nav = NavigationControl()
        self.velocity = 0.0
        self.omega = 0
        self.nav.publish_velocity(self.velocity, self.omega)

        rospy.on_shutdown(self.hook)


    def cbParametersChanged(self):
        self.publish_duration = rospy.Duration.from_sec(1.0 / self.process_frequency)
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = self.blobdetector_min_area
        params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)

    def detect_red_intersection(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red_ranges = {'lower': np.array([0, 150, 50]), 'upper': np.array([10, 255, 255])}
        

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
    
    def lane_detect(self, image_cv):
        # Crops and finds the lane; sets self.proportional
        crop = image_cv[300:, :, :]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        def find_largest(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_area = 20
            max_idx = -1
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_idx = i
            return contours, max_idx

        # Yellow mask first
        yellow_mask = cv2.inRange(crop_hsv, self.ROAD_MASK[0], self.ROAD_MASK[1])
        contours, idx = find_largest(yellow_mask)
        following_white = False
        mask_used = yellow_mask

        # Fallback: white lane
        if idx == -1:
            white_lower = np.array([120, 18, 155], np.uint8)
            white_upper = np.array([128, 39, 255], np.uint8)
            white_mask = cv2.inRange(crop_hsv, white_lower, white_upper)
            contours, idx = find_largest(white_mask)
            mask_used = white_mask
            following_white = True if idx != -1 else False

        # Compute centroid & proportional error
        if idx != -1:
            M = cv2.moments(contours[idx])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                offset = -(self.offset + 50) if following_white else self.offset
                self.proportional = cx - int(crop.shape[1] / 2) + offset

                # Debug draw
                if DEBUG_LANE_FOLLOW:
                    color = (255, 0, 0) if following_white else (0, 255, 0)
                    cv2.drawContours(crop, contours, idx, color, 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            else:
                self.proportional = None
        else:
            self.proportional = None

        # Publish debug mask
        if DEBUG_LANE_FOLLOW:
            debug_img = cv2.bitwise_and(crop, crop, mask=mask_used)
            msg_out = CompressedImage(
                format="jpeg", data=self.jpeg.encode(debug_img)
            )
            self.pub_mask.publish(msg_out)


    def detect_bot(self, image_cv):
        """
        Callback for processing a image which potentially contains a back pattern. Processes the image only if
        The pattern detection is performed using OpenCV's `findCirclesGrid <https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=solvepnp#findcirclesgrid>`_ function.

        Args:
            image_msg (:obj:`sensor_msgs.msg.CompressedImage`): Input image

        """
        
        (detection, centers) = cv2.findCirclesGrid(
            image_cv,
            patternSize=tuple(self.circlepattern_dims),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.simple_blob_detector,
        )

        if detection:
            xs = centers[:, 0, 0]  # Extract x coords
            pattern_width = np.max(xs) - np.min(xs)  # proxy for distance

            # Tail distance control: adjust based on how close the bot is
            target_width = 120   # desired pattern width when at target following distance
            error_distance = target_width - pattern_width

            # Turning adjustment based on horizontal position of pattern
            center_offset = np.mean(xs) - image_cv.shape[1] / 2  # deviation from center
            
            if DEBUG_TAIL and self.pub_circlepattern_image.get_num_connections() > 0:
                cv2.drawChessboardCorners(image_cv, tuple(self.circlepattern_dims), centers, detection)
                # Add text showing the width
                cv2.putText(
                    image_cv,
                    f"Width: {pattern_width:.1f}px",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                image_msg_out = self.bridge.cv2_to_compressed_imgmsg(image_cv)
                self.pub_circlepattern_image.publish(image_msg_out)

            return error_distance, center_offset
        
        return None
    
    def image_callback(self, msg):
        image_cv = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        now = rospy.Time.now()
        if now - self.last_stamp < self.publish_duration:
            return
        self.last_stamp = now

        # stopline_detected, distance = self.detect_red_intersection(image_cv)
        # if stopline_detected and distance < 30:
        #     self.nav.stop(3)

        # First check if bot was detected
        tail = self.detect_bot(image_cv)

        if tail is not None:
            self.tailing = True
            self.last_seen = rospy.Time.now()
            error_distance, offset = tail

            # Tuning parameters
            Kp_dist = 0.01
            Kp_angle = -0.05

            # Compute velocity and omega based on error
            v = Kp_dist * error_distance
            omega = Kp_angle * offset

            # Limit speed to avoid overshooting
            v = max(min(v, 0.35), 0.05) if v > 0 else 0

            rospy.loginfo(f"[Tailing] error={error_distance:.1f}, offset={offset:.1f} => v={v:.2f}, omega={omega:.2f}")
            # self.nav.publish_velocity(v, omega)
            self.nav.publish_velocity(0, 0)
            return
        
        # If bot hasn't been detected for tail_timout=0.5 seconds, continue lane following
        if (self.tailing and (now - self.last_seen) >= self.tail_timeout) or not self.tailing:
            self.tailing = False
            self.lane_detect(image_cv)

            if self.proportional is None:
                v = self.velocity
                omega = 0
                self.last_error = 0
                self.integral = 0
            else:
                current_time = rospy.get_time()
                dt = current_time - self.last_time
                if dt > 0:
                    d_error = (self.proportional - self.last_error) / dt
                    self.integral += self.proportional * dt
                else:
                    d_error = 0

                Pterm = -self.proportional * self.P
                Dterm = d_error * self.D
                Iterm = self.I * self.integral

                v = self.velocity
                omega = Pterm + Dterm + Iterm

                self.last_error = self.proportional
                self.last_time = current_time
        
            rospy.loginfo(f"[Lane Following] v={v:.2f}, omega={omega:.2f}")
            # self.nav.publish_velocity(v, omega)


    def hook(self):
        print("SHUTTING DOWN")
        for i in range(8):
            self.nav.publish_velocity(0,0)

if __name__ == '__main__':
    # create the node
    node = TailDuckNode(node_name='tail_duck_node')
    rospy.spin()