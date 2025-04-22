#!/usr/bin/env python3

import rospy
import os
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Range, CameraInfo
from geometry_msgs.msg import Point32
import cv2
from general_navigation import NavigationControl
from std_msgs.msg import ColorRGBA
from duckietown_msgs.msg import LEDPattern
from cv_bridge import CvBridge
from turbojpeg import TurboJPEG
import numpy as np

import dt_apriltags

DEBUG_LANE_FOLLOW = False
DEBUG_TAIL = False

class TailDuckNode(DTROS):
    def __init__(self, node_name):
        super(TailDuckNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']
        self.process_frequency = 5

        # --- Apriltag detection setup ---
        self.detector = dt_apriltags.Detector(families="tag36h11")
        self.tag_size = 0.065
        self.camera_parameters = None
        self.last_tag_id = -1
        camera_info_topic = f"/{self.veh}/camera_node/camera_info"
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback,  queue_size=1)
        self.tag_detection_pub = rospy.Publisher("/" + self.veh + '/tag_detections/compressed', CompressedImage, queue_size=1)

        

        # --- General collision prevention setup ---
        self.stop_bot = False
        self.tof_sub = rospy.Subscriber("/" + self.veh + "/front_center_tof_driver_node/range",
                                            Range,
                                            self.cbTOF,
                                            queue_size=1)

        # --- Duckiebot Tailing setup ---
        self.last_stamp = rospy.Time.now()
        self.circlepattern_dims = [7, 3]
        self.blobdetector_min_area = 10
        self.blobdetector_min_dist_between_blobs = 2
        self.cbParametersChanged()
        self.last_seen = rospy.Time(0)
        self.tail_timeout = rospy.Duration(0.65)
        self.last_tail_error = (0,0)
        self.last_offset = 0
        self.blue_direction = 'left'

        # --- Crosswalk and avoidance ----
        self.detect_crosswalks = True
        self.drive_dist = 0
        self.stop_time = 0
        self.maneuvering = False
        self.f_y = None
        self.f_c = None
        self.maneuver_state = 0
        self.state_time = 0
        self.detection_stage = 0
        self.stopped_at_crosswalk = False
        self.time_of_blue_stop = 0
        self.blue_cooldown_duration = 10
        

        # --- Lane following setup ---
        self.ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
        self.offset = 200
        self.P = 0.035
        self.D = -0.0033
        self.I = 0.000
        self.last_error = 0
        self.integral = 0
        self.last_time = rospy.get_time()

        # ---- Red intersection setup
        self.stopped_at_red = False
        self.time_of_red_stop = 0
        self.red_cooldown_duration = 10
        self.red_stops_count = 0

        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                        CompressedImage,
                        self.image_callback,
                        queue_size=1,
                        buff_size="20MB")
        self.pub_mask = rospy.Publisher(
            f"/{self.veh}/combined/mask/compressed", 
            CompressedImage, queue_size=1
        )
        self.pub_blue_debug = rospy.Publisher(
            f"/{self.veh}/debug/blue_bot_detection/compressed",
            CompressedImage, queue_size=1
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
        self.count = 0
        
        self.pub_leds = rospy.Publisher(f"/{self.veh}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)

        self.nav = NavigationControl()
        self.velocity = 0.3
        self.omega = 0
        self.nav.publish_velocity(0, self.omega)

        rospy.on_shutdown(self.hook)

    # def init_image_sub(self):
    #     self.camera_info_sub = rospy.Subscriber(
    #         self.img_topic,
    #         CompressedImage,
    #         self.camera_callback,
    #         queue_size=1
    #     )

    def camera_info_callback(self, msg):
        self.camera_calibration = msg

        # print("== Calibrating Camera ==")

        # currRawImage_height = img.shape[0]
        # currRawImage_width = img.shape[1]
        currRawImage_height = 640
        currRawImage_width = 480

        scale_matrix = np.ones(9)
        if self.camera_calibration.height != currRawImage_height or self.camera_calibration.width != currRawImage_width:
            scale_width = float(currRawImage_width) / self.camera_calibration.width
            scale_height = float(currRawImage_height) / self.camera_calibration.height
            scale_matrix[0] *= scale_width
            scale_matrix[2] *= scale_width
            scale_matrix[4] *= scale_height
            scale_matrix[5] *= scale_height

        self.tag_size = 0.065 #rospy.get_param("~tag_size", 0.065)
        rect_K, _ = cv2.getOptimalNewCameraMatrix(
            (np.array(self.camera_calibration.K)*scale_matrix).reshape((3, 3)),
            self.camera_calibration.D,
            (640,480),
            1.0
        )
        self.camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])


        try:
            self.subscriberCameraInfo.shutdown()
            self.safeToRunProgram = True
            # print("== Camera Info Subscriber successfully killed ==")
        except BaseException:
            pass
        
        # paramenters for estimating the ground distance for detecting objects
        self.f_y = rect_K[1, 1]
        self.c_y = rect_K[1, 2]



    def detect_apriltag(self, image_cv):
        """
        Detects apriltags in the image and returns the tag ID and its position.
        """
        if self.camera_parameters is None:
            return None
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)
        closest_tag_id = 0
        closest = 0

        if len(tags) == 0:
            self.dist_from_april = 999/2
            self.error_from_april = 0

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', image_cv)[1]).tobytes()
            self.tag_detection_pub.publish(msg)
            return None

        if len(tags) > 0:
            for tag in tags:
                (ptA, ptB, ptC, ptD) = tag.corners
                diff = abs(ptA[0] - ptB[0])
                tag_id = tag.tag_id
                (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
                txt_col = (25, 25, 200)
                cv2.putText(image_cv, str(tag_id), (cX - 9, cY + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_col, 2)
                if diff > closest:
                    closest = diff
                    closest_tag_id = tag.tag_id

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', image_cv)[1]).tobytes()
            self.tag_detection_pub.publish(msg)
            

        return int(closest_tag_id)

    def stop_at_red(self):
        if not self.stopped_at_red:
            rospy.loginfo("Stopping at red")
            self.time_of_red_stop = rospy.get_time()
            self.nav.stop(3)
            self.stopped_at_red = True

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

    def cbTOF(self, msg):
        if 0.05 < msg.range <= 0.2:
            # rospy.loginfo(f"Detected object at : {msg.range}")
            self.stop_bot = True
        else:
            self.stop_bot = False

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
        # rospy.loginfo(f"CircleGrid dt={dt:.1f}ms, found={found}")

        # Prepare a debug copy
        debug = image_cv.copy()

        if found:
            # compute width + offset
            xs = centers[:, 0, 0]
            pattern_width = float(np.max(xs) - np.min(xs))
            error_distance = 100.0 - pattern_width
            center_offset  = float(np.mean(xs) - (image_cv.shape[1] / 2))
            self.last_offset = center_offset

            # annotate
            cv2.drawChessboardCorners(debug, tuple(self.circlepattern_dims), centers, found)
            cv2.putText(
                debug, f"W: {pattern_width:.1f}px, last_offser: {self.last_offset}",
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
                # rospy.loginfo("Not detected")

        # --- 3) Always publish debug image ---
        # if DEBUG_TAIL:
        imgmsg = self.bridge.cv2_to_compressed_imgmsg(debug)
        self.pub_circlepattern_image.publish(imgmsg)

        return result
    
    def detect_blue_bot(self, image_cv):
        """
        Detects the blue trailing bot in the image and returns "left"/"right" 
        based on its position. Also publishes a debug image with the contour overlaid.
        """
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([106, 68, 0])
        upper_blue = np.array([151, 255, 145])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = image_cv.copy()

        direction = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                blue_center = x + w // 2

                # draw a box around the detected bot
                cv2.rectangle(debug, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # draw center line
                cv2.line(debug,
                        (blue_center, 0),
                        (blue_center, debug.shape[0]),
                        (255, 0, 0), 1)

                if blue_center < debug.shape[1] // 2:
                    direction = "left"
                    cv2.putText(debug, "LEFT", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                else:
                    direction = "right"
                    cv2.putText(debug, "RIGHT", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # publish debug image
        if DEBUG_TAIL:
            blue_dbg_msg = self.bridge.cv2_to_compressed_imgmsg(debug)
            self.pub_blue_debug.publish(blue_dbg_msg)

        return direction


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
            white_lower = np.array([0, 0, 180], np.uint8)
            white_upper = np.array([180, 60, 255], np.uint8)
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

                offset = -(self.offset + 90) if following_white else self.offset
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

    def detect_crosswalk(self, image):
        """
        Detects the crosswalks
        """
        # self.time_of_blue_stop = rospy.get_time()
        # self.stopped_at_crosswalk = True

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
        """
        Detects duckpedestrians.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        duck_ranges = {'lower': np.array([9, 91, 163]), 'upper': np.array([22, 255, 255])}
    
        mask = cv2.inRange(hsv, duck_ranges['lower'], duck_ranges['upper'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.contourArea(largest_contour) > 500

    def detect_broken_bot(self, image_cv):
        """
        Detects the broken bot in the image and returns if the bot was detected (bool), which triggers maneuver, and the estimated (ground) distance in cm to the bot
        Also publishes a debug image with the contour overlaid.
        """
        # Crop top part of the image to reduce false positives
        crop_offset = 260
        cropped_img = image_cv[crop_offset:, :]

        # Convert to HSV and mask for blue
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([106, 68, 50])
        upper_blue = np.array([151, 255, 200])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = cropped_img.copy()

        broken_bot_detected = False
        # dist_cm = float('inf')
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                blue_center = x + w // 2

                # # -- to distinguish between crosswalk and bot
                # aspect_ratio = w / h
                # image_height = cropped_img.shape[0]
                # # Filter: shape and position
                # if aspect_ratio > 2.0:
                #     rospy.loginfo("[BrokenBot] Skipped — likely crosswalk due to aspect ratio")
                #     return False, float('inf')

                # if y + h > image_height - 40:
                #     rospy.loginfo("[BrokenBot] Skipped — detection too low, likely crosswalk")
                #     return False, float('inf')
                # # --

                # y_bot = y + h
                # y_img = y_bot + crop_offset

                # if self.f_y is not None and self.c_y is not None:
                #     cam_height_cm = 10
                #     dist_cm = (cam_height_cm * self.f_y) / (y_img - self.c_y)

                # draw a box around the detected bot
                cv2.rectangle(debug, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # draw center line
                cv2.line(debug,
                        (blue_center, 0),
                        (blue_center, debug.shape[0]),
                        (255, 0, 0), 1)
                # # annotate distance
                # cv2.putText(debug, f"{dist_cm:.1f}cm",
                #         (x, y - 10),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.5,
                #         (255, 255, 255), 1)
                
                broken_bot_detected = True

        # publish debug image
        # if DEBUG_TAIL:
        blue_dbg_msg = self.bridge.cv2_to_compressed_imgmsg(debug)
        self.pub_blue_debug.publish(blue_dbg_msg)

        return broken_bot_detected #, dist_cm

    def maneuver_around_bot(self):
        """
        Maneuvers around (to the left) the broken bot in the image
        """
        self.state_time += 1
        turn_angle = 2.5 #rad/sec
        turn_time = 10 #~1second
        straight_time = 12 #~5seconds
        if self.state_time < 5:
            return 0, 0
        if self.maneuver_state == 0:
            # Wait before turning
            if self.state_time > 10:
                self.maneuver_state += 1
                self.state_time = 0
            return 0, 0
        elif self.maneuver_state == 1:
            # First turn (left) to shift into next lane
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            # return -0.25, turn_angle
            return 0, turn_angle
        elif self.maneuver_state == 2:
            # Drive forward into the new lane
            if self.state_time > straight_time - 3:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, 0
        elif self.maneuver_state == 3:
            # Turn back into original direction
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            # return 0.25, -turn_angle
            return 0, -turn_angle
        elif self.maneuver_state == 4:
            # Continue driving to pass the broken bot
            if self.state_time > straight_time + 5:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, 0
        elif self.maneuver_state == 5:
            # Start retunring to the original lane
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0, -turn_angle+1.0
        # elif self.maneuver_state == 6:
        #     # Move straight back towards center
        #     if self.state_time > straight_time-15:
        #         self.maneuver_state += 1
        #         self.state_time = 0
        #     return 0.25, 0
        # elif self.maneuver_state == 7:
        #     # Final alignement with original direction
        #     if self.state_time > turn_time:
        #         self.maneuver_state += 1
        #         self.state_time = 0
        #     return 0, turn_angle
        else:
            # Maneuver complete
            self.detection_stage = 2
            self.maneuvering = False
            self.state_time = 0
            self.maneuver_state = 0
            return 0.2, 0
    
    def maneuver_around_bot_copy(self):
        """
        Maneuvers around (to the left) the broken bot in the image
        """
        self.state_time += 1
        turn_angle = 2.2 #rad/sec
        turn_time = 25 #~1second
        straight_time = 15 #~5seconds
        if self.state_time < 5:
            return 0, 0
        if self.maneuver_state == 0:
            # Wait before turning
            if self.state_time > 10:
                self.maneuver_state += 1
                self.state_time = 0
            return 0, 0
        elif self.maneuver_state == 1:
            # First turn (left) 45 degrees into next lane
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            # return -0.25, turn_angle
            return 0, turn_angle
        elif self.maneuver_state == 2:
            # Drive forward into the new lane beside/past broken bot
            if self.state_time > straight_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, 0
        elif self.maneuver_state == 3:
            # Turn back 45 degrees to face original lane
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
            # return 0.25, -turn_angle
            return 0, -turn_angle
        elif self.maneuver_state == 4:
            # Drive back into original lane
            if self.state_time > straight_time:
                self.maneuver_state += 1
                self.state_time = 0
            return 0.25, 0
        # elif self.maneuver_state == 5:
        #     # Start retunring to the original lane
        #     if self.state_time > turn_time:
        #         self.maneuver_state += 1
        #         self.state_time = 0
        #     return 0, -turn_angle+1.0
        # elif self.maneuver_state == 6:
        #     # Move straight back towards center
        #     if self.state_time > straight_time-15:
        #         self.maneuver_state += 1
        #         self.state_time = 0
        #     return 0.25, 0
        # elif self.maneuver_state == 7:
        #     # Final alignement with original direction
        #     if self.state_time > turn_time:
        #         self.maneuver_state += 1
        #         self.state_time = 0
        #     return 0, turn_angle
        else:
            # Maneuver complete
            self.detection_stage = 2
            self.maneuvering = False
            self.state_time = 0
            self.maneuver_state = 0
            return 0.2, 0

###################################################################################################################

    def image_callback(self, msg):
        image_cv = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        now = rospy.Time.now()
        
        if self.count<1:
            self.set_led_color(self.light_color_list)
            # self.init_image_sub()
            self.count = 1

        if now - self.last_stamp < self.publish_duration:
            return
        self.last_stamp = now

        # ------------------- Manuever broken bot logic -------------------------
        if self.maneuvering:
            vel, omega = self.maneuver_around_bot_copy()
            self.nav.publish_velocity(vel, omega)
            return

        broken_bot = self.detect_broken_bot(image_cv)
        if broken_bot and self.detection_stage == 1:
            rospy.loginfo(f"Broken bot detected — initiating maneuver")
            self.maneuvering = True
            vel, omega = self.maneuver_around_bot_copy()
            self.nav.publish_velocity(vel, omega)
            return
        else:
            if self.stop_bot:
                self.nav.publish_velocity(0,0)
                return
        # ------------------------------------------------------------------------

        # -------------------- Crosswalk logic -----------------------------------
        # if self.red_stops_count => 5:
        rospy.loginfo(self.detection_stage)
        if self.detection_stage in [0, 2]: 
            if self.detect_crosswalks or self.stop_time < 10:
                stopwalk_detection, blue_distance = self.detect_crosswalk(image_cv)
                if stopwalk_detection and blue_distance < 25:
                    rospy.loginfo("Crosswalk detected - stopping...")
                    self.detect_crosswalks = False
                    vel = 0
                    self.nav.publish_velocity(vel, 0)
                    self.stop_time += 1
                    self.blue_cooldown_duration = 15
                    return
            else:
                if self.detect_ducks(image_cv):
                    rospy.loginfo("Ducks detected - waiting...")
                    vel = 0
                    self.nav.publish_velocity(vel, 0)
                    return
                elif self.drive_dist < 10:
                    rospy.loginfo("Driving through crosswalk")
                    # vel = 0.5
                    self.drive_dist += 1
                else:
                    rospy.loginfo("Crosswalk complete - resuming lane following...")
                    self.detection_stage += 1
                    self.detect_crosswalks = True
                    self.drive_dist = 0
                    self.stop_time = 0
                    self.blue_cooldown_duration = 15
                    self.stopped_at_crosswalk = False
        # -----------------------------------------------------------------------------

        # # ------------------- Manuever broken bot logic -----------------------
        # broken_bot, dist = self.detect_broken_bot(image_cv)
        # # if self.red_stops_count => 5:
        # if self.detection_stage==1 and ((broken_bot and dist < 10) or self.maneuvering) and ((rospy.get_time() - self.time_of_blue_stop) > self.blue_cooldown_duration):
        #     if self.maneuvering:
        #         vel, omega = self.maneuver_around_bot()
        #         self.nav.publish_velocity(vel, omega)
        #         return
        #     elif self.detection_stage == 1: 
        #         rospy.loginfo("Checking for broken bot...")
        #         if broken_bot and dist < 10:
        #             rospy.loginfo(f"Broken bot detected at {dist:.1f}cm — initiating maneuver")
        #             self.maneuvering = True
        #             vel, omega = self.maneuver_around_bot()
        #             self.nav.publish_velocity(vel, omega)
        #             return
        #     else: # Only stop, no maneuver and no detect broken bot
        #         # rospy.loginfo("stop Callback")
        #         self.nav.publish_velocity(0,0)
        #         return
        # #else:
        #     # if self.stop_bot:
        #     #     self.nav.publish_velocity(0,0)
        #     #     return

        # # Detect the apriltag in the image
        # # tag_id = None
        # tag_id = self.detect_apriltag(image_cv)
        
        # # Always stop at red if not stopped already
        # stopline_detected, distance = self.detect_red_intersection(image_cv)

        # if stopline_detected and distance < 50:
        #     blue_direction = self.detect_blue_bot(image_cv)
        #     if blue_direction is not None:
        #         self.blue_direction = blue_direction

        # if stopline_detected and distance < 20 and (rospy.get_time() - self.time_of_red_stop) > self.red_cooldown_duration:
        #     self.stop_at_red()
        #     self.stopped_at_red = False
        #     rospy.loginfo(self.time_of_red_stop)

        #     if self.red_stops_count == 0:
        #         if self.blue_direction == "left":
        #             self.nav.turn_left(0.4, 2.0, extra=0.9)
        #             rospy.loginfo("Left turn")
        #         elif self.blue_direction == "right":
        #             self.nav.move_straight(0.35)
        #             self.nav.turn_right(0, -2.5, extra=0.5)
        #             rospy.loginfo("Right turn")
        #         self.red_stops_count += 1

        #     elif self.red_stops_count == 1:
        #         self.nav.move_straight(0.5)
        #         rospy.loginfo("Straight")
        #         self.red_stops_count += 1
        #         self.red_cooldown_duration = 15 # JUst so it doesn't detect red while going straight

            # elif self.red_stops_count == 2:
            #     if self.blue_direction == "left":
            #         self.nav.turn_left(0.35, 2.0, extra=0.8)
            #         rospy.loginfo("Left turn")
            #     elif self.blue_direction == "right":
            #         self.nav.move_straight(0.35)
            #         self.nav.turn_right(0, -2.5)
            #         rospy.loginfo("Right turn")
            #     self.red_stops_count += 1
            #     self.red_cooldown_duration = 10

            # elif self.red_stops_count == 3:
            #     # logic if a tag was seen
            #     if tag_id is not None:
            #         if tag_id == 48:
            #             rospy.loginfo("Turning LEFT at AprilTag 21")
            #             self.nav.turn_left()
            #         elif tag_id == 50:
            #             rospy.loginfo("Turning RIGHT at AprilTag 59")
            #             self.nav.turn_right()
            #         else:
            #             rospy.logwarn(f"Unknown tag ID: {tag_id}")
            #     else:
            #         rospy.loginfo("No tag seen. Proceeding forward.")
            #     self.red_stops_count += 1

            # elif self.red_stops_count == 4:
            #     # logic if a tag was seen
            #     if tag_id is not None:
            #         if tag_id == 48:
            #             rospy.loginfo("Turning LEFT at AprilTag 21")
            #             self.nav.turn_left()
            #         elif tag_id == 50:
            #             rospy.loginfo("Turning RIGHT at AprilTag 59")
            #             self.nav.turn_right()
            #         else:
            #             rospy.logwarn(f"Unknown tag ID: {tag_id}")
            #     else:
            #         rospy.loginfo("No tag seen. Proceeding forward.")
            #     self.red_stops_count += 1

            # elif self.red_stops_count == 5:
            #     # TODO: Stage 4 Parking
            #     pass

            # rospy.loginfo(self.red_stops_count)
        
        # # # ------------------- Crosswalk logic ----------------------------------
        # # # if self.red_stops_count => 5:
        # rospy.loginfo(self.detection_stage)
        # if self.detection_stage in [0, 2]: 
        #     if self.detect_crosswalks or self.stop_time < 10:
        #         stopwalk_detection, _ = self.detect_crosswalk(image_cv)
        #         if stopwalk_detection:
        #             rospy.loginfo("Crosswalk detected - stopping...")
        #             self.detect_crosswalks = False
        #             vel = 0
        #             self.nav.publish_velocity(vel, 0)
        #             self.stop_time += 1
        #             self.blue_cooldown_duration = 15
        #             return
        #     else:
        #         if self.detect_ducks(image_cv):
        #             rospy.loginfo("Ducks detected - waiting...")
        #             vel = 0
        #             self.nav.publish_velocity(vel, 0)
        #             return
        #         elif self.drive_dist < 10:
        #             rospy.loginfo("Driving through crosswalk")
        #             # vel = 0.5
        #             self.drive_dist += 1
        #         else:
        #             rospy.loginfo("Crosswalk complete - resuming lane following...")
        #             self.detection_stage += 1
        #             self.detect_crosswalks = True
        #             self.drive_dist = 0
        #             self.stop_time = 0
        #             self.blue_cooldown_duration = 15
        #             self.stopped_at_crosswalk = False
            
        # lane-follow if bot not seen
        if ((now - self.last_seen) >= self.tail_timeout):
            # self.tailing = False
            self.lane_detect(image_cv)

            if self.proportional is None:
                # v = self.velocity
                self.omega = 0
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

                # v = self.velocity
                self.omega = Pterm + Dterm + Iterm

                self.last_error = self.proportional
                self.last_time = current_time
            # rospy.loginfo(f"[Lane Following] v={self.velocity:.2f}, omega={self.omega:.2f}")
            self.nav.publish_velocity(self.velocity, self.omega)

        # tail = self.detect_bot(image_cv)
        # if tail is not None:
        #     # self.set_led_color([
        #     #                     [0, 0, 0, 0],
        #     #                     [0, 0, 1, 0],
        #     #                     [0, 0, 0, 0],
        #     #                     [0, 0, 1, 1],
        #     #                     [0, 0, 0, 1],
        #     #                      ])
        #     error_distance, offset = tail

        #     # Tuning parameters
        #     Kp_dist = 0.013
        #     Kp_angle = -0.005

        #     # Compute velocity and omega based on error
        #     v = Kp_dist * error_distance
        #     omega = Kp_angle * offset

        #     # Limit speed to avoid overshooting
        #     v = max(min(v, 0.3), 0.05) if v > 0 else 0

        #     # Preventing collision is highest priority

        #     # rospy.loginfo(f"[Tailing] error={error_distance:.1f}, offset={offset:.1f} => v={v:.2f}, omega={omega:.2f}")
        #     self.nav.publish_velocity(v, omega)
        #     return

    def hook(self):
        print("SHUTTING DOWN")
        for i in range(8):
            self.nav.publish_velocity(0,0)

if __name__ == '__main__':
    # create the node
    node = TailDuckNode(node_name='tail_duck_node')
    rospy.spin()