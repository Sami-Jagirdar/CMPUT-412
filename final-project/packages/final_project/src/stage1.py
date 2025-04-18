#!/usr/bin/env python3

import rospy
import os
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point32
import cv2
from general_navigation import NavigationControl
from std_msgs.msg import ColorRGBA
from duckietown_msgs.msg import LEDPattern
from cv_bridge import CvBridge
from turbojpeg import TurboJPEG
import numpy as np

DEBUG_LANE_FOLLOW = True
DEBUG_TAIL = False

class TailDuckNode(DTROS):
    def __init__(self, node_name):
        super(TailDuckNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']
        self.process_frequency = 5

        # --- Duckiebot Tailing setup ---
        self.last_stamp = rospy.Time.now()
        self.circlepattern_dims = [7, 3]
        self.blobdetector_min_area = 10
        self.blobdetector_min_dist_between_blobs = 2
        self.cbParametersChanged()
        self.last_seen = rospy.Time(0)
        self.tail_timeout = rospy.Duration(0.65)
        self.tailing = False
        self.last_tail_error = (0,0)
        self.target_width  = 120.0

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

        self.pub_circlepattern_image = rospy.Publisher("/{}/duckiebot_detection_node/detection_image/compressed".format(os.environ['VEHICLE_NAME']), CompressedImage, queue_size=1)
        self.log("Detection Initialization completed.")

        self.LEDspattern = LEDPattern()
        self.light_color_list = [ # init led lights
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                 ]
        
        self.pub_leds = rospy.Publisher(f"/{self.veh}/led_emitter_node/led_pattern", LEDPattern, queue_size=10)

        self.nav = NavigationControl()
        self.velocity = 0.3
        self.omega = 0
        self.nav.publish_velocity(self.velocity, self.omega)

        rospy.on_shutdown(self.hook)

    def publish_LED_pattern(self):
        # Publish the LED pattern to the led_emitter_node
        self.LEDspattern.rgb_vals = []
        for i in range(5):
            rgba = ColorRGBA()
            rgba.r = self.light_color_list[i][0]
            rgba.g = self.light_color_list[i][1]
            rgba.b = self.light_color_list[i][2]
            rgba.a = self.light_color_list[i][3]

            self.LEDspattern.rgb_vals.append(rgba)
        self.pub_leds.publish(self.LEDspattern)

    def set_led_color(self, colors):
        # Set the color of the LEDs

        # colors should be a list of length 5 with 
        # each element being a list of length 4
        for i in range(len(self.light_color_list)):
            if len(colors[i])==3:
                self.light_color_list[i] = colors[i] + [1]
            else:
                self.light_color_list[i] = colors[i]
        
        self.publish_LED_pattern()

    def cbParametersChanged(self):
        self.publish_duration = rospy.Duration.from_sec(1.0 / self.process_frequency)

        # 1) Improve contrast: CLAHE on the gray channel
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 2) Blob detector tuning
        params = cv2.SimpleBlobDetector_Params()

        # Thresholding: scan from dark to bright
        params.minThreshold = 10
        params.maxThreshold = 200
        params.thresholdStep = 10

        # Keep only roughly circle‑shaped blobs
        params.filterByArea = True
        params.minArea = 20           # increase this if you get too many tiny blobs
        params.maxArea = 5000         # decrease if you pick up big glare patches

        params.filterByCircularity = True
        params.minCircularity = 0.7

        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        params.filterByConvexity = True
        params.minConvexity = 0.8

        # (Optionally) filter by color if your dots are always dark or always light
        # params.filterByColor = True
        # params.blobColor = 0

        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)
    
    def detect_bot(self, image_cv):
        """
        Runs a oneshot grid detection on a prefiltered image, logs timing, and
        always publishes a debug view showing either the corners+width or "No pattern".
        """
        # self.set_led_color(self.light_color_list)

        # --- 1) Pre‑process ---
        #  a) Gaussian blur to smooth noise
        blurred = cv2.GaussianBlur(image_cv, (5, 5), 0)
        #  b) Equalize the V channel for consistent contrast
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)

        # --- 2) Detect & time it ---
        t0 = rospy.get_time()
        flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
        found, centers = cv2.findCirclesGrid(
            gray,
            patternSize=tuple(self.circlepattern_dims),
            flags=flags,
            blobDetector=self.simple_blob_detector,
        )
        dt = (rospy.get_time() - t0) * 1000
        rospy.loginfo(f"CircleGrid dt={dt:.1f}ms, found={found}")

        # Prepare a debug copy
        debug = image_cv.copy()

        if found:

            # compute width + offset
            xs = centers[:, 0, 0]
            pattern_width = float(np.max(xs) - np.min(xs))
            error_distance = 120.0 - pattern_width
            center_offset  = float(np.mean(xs) - (image_cv.shape[1] / 2))

            # annotate
            cv2.drawChessboardCorners(debug, tuple(self.circlepattern_dims), centers, found)
            cv2.putText(
                debug, f"W: {pattern_width:.1f}px",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2, cv2.LINE_AA
            )

            # update last‑seen for your hold logic
            self.last_seen    = rospy.Time.now()
            self.last_pattern = (error_distance, center_offset)

            result = (error_distance, center_offset)

        else:
            # annotate “no pattern”
            cv2.putText(
                debug, "No pattern",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2, cv2.LINE_AA
            )

            # if you want to hold the last good result up to tail_timeout:
            if (rospy.Time.now() - self.last_seen) < self.tail_timeout:
                result = self.last_pattern
            else:
                result = None

        # --- 3) Always publish debug image ---
        imgmsg = self.bridge.cv2_to_compressed_imgmsg(debug)
        self.pub_circlepattern_image.publish(imgmsg)

        return result

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

    def image_callback(self, msg):
        image_cv = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        now = rospy.Time.now()
        if now - self.last_stamp < self.publish_duration:
            return
        self.last_stamp = now

        # stopline_detected, distance = self.detect_red_intersection(image_cv)
        # if stopline_detected and distance < 30:
        #     self.nav.stop(3)

        # lane-following takes precedence
        # if (self.tailing and (now - self.last_seen) >= self.tail_timeout) or not self.tailing:
        #     self.tailing = False
        #     self.lane_detect(image_cv)

        #     if self.proportional is None:
        #         # v = self.velocity
        #         self.omega = 0
        #         self.last_error = 0
        #         self.integral = 0
        #     else:
        #         current_time = rospy.get_time()
        #         dt = current_time - self.last_time
        #         if dt > 0:
        #             d_error = (self.proportional - self.last_error) / dt
        #             self.integral += self.proportional * dt
        #         else:
        #             d_error = 0

        #         Pterm = -self.proportional * self.P
        #         Dterm = d_error * self.D
        #         Iterm = self.I * self.integral

        #         # v = self.velocity
        #         self.omega = Pterm + Dterm + Iterm

        #         self.last_error = self.proportional
        #         self.last_time = current_time

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
            return
        
        rospy.loginfo(f"[Lane Following] v={self.velocity:.2f}, omega={self.omega:.2f}")
        # self.nav.publish_velocity(self.velocity, self.omega)
        self.nav.publish_velocity(0,0)



    def hook(self):
        print("SHUTTING DOWN")
        for i in range(8):
            self.nav.publish_velocity(0,0)

if __name__ == '__main__':
    # create the node
    node = TailDuckNode(node_name='tail_duck_node')
    rospy.spin()