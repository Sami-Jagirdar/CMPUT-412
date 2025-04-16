#!/usr/bin/env python3

# 1.1 - 1.3 and 2.1

# import required libraries
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
import numpy as np
from computer_vision.srv import GetLaneInfo, GetLaneInfoResponse
from computer_vision.msg import LaneDistance
import cv2
from cv_bridge import CvBridge

RED=1
GREEN=2
BLUE=3
YELLOW=4
WHITE=5

class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        # add your code here
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self._distorted_img_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._undistorted_img_topic = f"/{self._vehicle_name}/camera_node/undistorted_image/compressed"
        self._color_detection_topic = f"/{self._vehicle_name}/lane_detection_node/color_detection/compressed"

        # camera calibration parameters (intrinsic matrix and distortion coefficients)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False
        self.mapx = None
        self.mapy = None
        self.roi = None
        self.camera_info_sub = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)

        # color detection parameters in HSV format
        self.red_lower = np.array([0,95,108], np.uint8)
        self.red_upper = np.array([8,171,255], np.uint8)
        self.green_lower = np.array([44,56,140], np.uint8)
        self.green_upper = np.array([102,106,188], np.uint8)
        self.blue_lower = np.array([85,110,108], np.uint8)
        self.blue_upper = np.array([114,238,210], np.uint8)

        # w,h of each lane
        self.detected_lanes = {
        'red': {'detected': False, 'contour': None, 'dimensions': None, 'bbox': None},
        'green': {'detected': False, 'contour': None, 'dimensions': None, 'bbox': None},
        'blue': {'detected': False, 'contour': None, 'dimensions': None, 'bbox': None},
        'yellow': {'detected': False, 'contour': None, 'dimensions': None, 'bbox': None},
        'white': {'detected': False, 'contour': None, 'dimensions': None, 'bbox': None}
        }

        # initialize bridge and subscribe to camera feed
        self.bridge = CvBridge()
        self.camera_sub = rospy.Subscriber(self._distorted_img_topic, CompressedImage, self.callback)

        # lane detection publishers
        self.undistorted_pub = rospy.Publisher(self._undistorted_img_topic, CompressedImage, queue_size=1)
        self.color_detection_pub = rospy.Publisher(self._color_detection_topic, CompressedImage, queue_size=1)

        # These lane detection topic messages used for lane following  
        self._lane_detection_topic = f"/{self._vehicle_name}/lane_detection_node/lane_info"
        self.lane_pub = rospy.Publisher(self._lane_detection_topic, LaneDistance, queue_size=1)

        # self._yellow_lane_distance_topic = f"/{self._vehicle_name}/lane_detection_node/yellow_lane_distance"
        # self._white_lane_distance_topic = f"/{self._vehicle_name}/lane_detection_node/white_lane_distance"
        # self.yellow_lane_distance_pub = rospy.Publisher(self._yellow_lane_distance_topic, LaneDistance, queue_size=1)
        # self.white_lane_distance_pub = rospy.Publisher(self._white_lane_distance_topic, LaneDistance, queue_size=1)

        # Homomgraphy initalization for estimating distance to lane
        # The homography parameters were exported from the duckiebot dashboard after performing extrinsic calibration
        homography = [
        -4.99621433668091e-05, 0.0012704090688819693, 0.2428235605203261,
        -0.001999628080487182, -5.849807527639727e-05, 0.6400119336043912,
        0.0003409556379103712, 0.0174415825291776, -3.2316507961510252
        ]

        # # Homography of a different bot
        # homography = [-4.4176150901508024e-05, 0.0004846965281425196, 0.30386285736827856,
        #             -0.0015921548874079563, 1.0986376221035314e-05, 0.4623061006515125,
        #             -0.00029009271219456394, 0.011531091762722323, -1.4589875279686733
        # ]
        
        self.H = np.array(homography).reshape((3,3))

        self.org_img_w = 640
        self.org_img_h = 480
        
        # ROI vertices
        self.roi = None

        # Set up services for PART 1 lane detection
        self.lane_info_service = rospy.Service(
            f'/{self._vehicle_name}/{node_name}/get_lane_info', 
            GetLaneInfo, 
            self.get_lane_info
        )
        # define other variables as needed
        self.rate = rospy.Rate(20)

    def camera_info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.K).reshape((3,3))
            self.dist_coeffs = np.array(msg.D)

            rospy.loginfo(f"Camera Intrinsic Matrix: \n{self.camera_matrix}\n")
            rospy.loginfo(f"Distortion Coefficients: \n{self.dist_coeffs}\n")
            
            self.camera_info_sub.unregister()

            #Find the new Optimal Camera matrix
            h,w = 480, 640 # Image shape was found from exercise 2
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs,
                                                                   (w,h), 1, (w,h))
            
            # Find the undistortion mapping matrix to be used for all images
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, 
                                                               None, new_camera_matrix, (w,h), cv2.CV_32FC1) #cv2.CV_32FC1=5

            self.roi = roi
            self.got_camera_info = True

    def undistort_image(self, image):
        # add your code here
        undistorted_image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        x,y,w,h = self.roi
        return undistorted_image[y:y+h, x:x+w]

    def preprocess_image(self, image):
        # add your code here
        resized_image = cv2.resize(image, (320, 240))
        h,_ = resized_image.shape[:2]
        cropped_image = resized_image[h//2:,:]
        return cv2.GaussianBlur(cropped_image, (5,5), 0)
    
    
    def detect_lane_color(self, image):
        # add your code here
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_mask = cv2.inRange(hsvFrame, self.red_lower, self.red_upper)

        green_mask = cv2.inRange(hsvFrame, self.green_lower, self.green_upper)

        blue_mask = cv2.inRange(hsvFrame, self.blue_lower, self.blue_upper)

        kernel = np.ones((5,5), "uint8")

        # For red color 
        red_mask = cv2.dilate(red_mask, kernel) 
        res_red = cv2.bitwise_and(image, image, 
                                mask = red_mask) 
        
        # For green color 
        green_mask = cv2.dilate(green_mask, kernel) 
        res_green = cv2.bitwise_and(image, image, 
                                    mask = green_mask) 
        
        # For blue color 
        blue_mask = cv2.dilate(blue_mask, kernel) 
        res_blue = cv2.bitwise_and(image, image, 
                                mask = blue_mask) 

        # Creating contour to track red color 
        contours, hierarchy = cv2.findContours(red_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                self.detected_lanes['red']['detected'] = True
                self.detected_lanes['red']['contour'] = contour
                self.detected_lanes['red']['dimensions'] = (w, h)
                self.detected_lanes['red']['bbox'] = (x, y, w, h)
                bottom_center_x = x + w // 2
                bottom_center_y = y + h + self.org_img_h // 4
                ground_x, ground_y = self.project_pixel_to_ground(bottom_center_x*2, bottom_center_y*2)
                image = cv2.rectangle(image, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2)
                dimensions_text = f"{w}x{h},{ground_x:0.2f}m"
                cv2.putText(image, dimensions_text, (x, y + h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)     

        # Creating contour to track green color 
        contours, hierarchy = cv2.findContours(green_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
    
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour)
                self.detected_lanes['green']['detected'] = True
                self.detected_lanes['green']['contour'] = contour
                self.detected_lanes['green']['dimensions'] = (w, h)
                self.detected_lanes['green']['bbox'] = (x, y, w, h)
                bottom_center_x = x + w // 2
                bottom_center_y = y + h + self.org_img_h // 4
                ground_x, ground_y = self.project_pixel_to_ground(bottom_center_x*2, bottom_center_y*2)
                image = cv2.rectangle(image, (x, y), 
                                        (x + w, y + h), 
                                        (0, 255, 0), 2) 
                dimensions_text = f"{w}x{h},{ground_x:0.2f}m"
                cv2.putText(image, dimensions_text, (x, y + h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Creating contour to track blue color 
        contours, hierarchy = cv2.findContours(blue_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour)
                self.detected_lanes['blue']['detected'] = True
                self.detected_lanes['blue']['contour'] = contour
                self.detected_lanes['blue']['dimensions'] = (w, h)
                self.detected_lanes['blue']['bbox'] = (x, y, w, h)
                bottom_center_x = x + w // 2
                bottom_center_y = y + h + self.org_img_h // 4
                ground_x, ground_y = self.project_pixel_to_ground(bottom_center_x*2, bottom_center_y*2)
                image = cv2.rectangle(image, (x, y), 
                                        (x + w, y + h), 
                                        (255, 0, 0), 2) 
                dimensions_text = f"{w}x{h}.{ground_x:0.2f}m"
                cv2.putText(image, dimensions_text, (x, y + h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        return image


    def project_pixel_to_ground(self, pixel_x, pixel_y):

        point = np.array([pixel_x, pixel_y, 1.0]).reshape(3,1)
        ground_point = np.matmul(self.H, point).reshape(3)

        if abs(ground_point[2]) > 1e-7:
            ground_x = ground_point[0] / ground_point[2]
            ground_y = ground_point[1] / ground_point[2]
            return ground_x, ground_y
        else:
            rospy.logwarn("Point is too close to judge distance, zero division error")
            return None, None
        
    # Service Callback
    def get_lane_info(self, req):
        response = GetLaneInfoResponse()
        
        # Map the color enum to the corresponding lane key
        color_map = {
            RED: 'red',
            GREEN: 'green',
            BLUE: 'blue'
        }
        
        # Check if the requested color is valid
        if req.color not in color_map:
            rospy.logwarn(f"Invalid color requested: {req.color}")
            response.detected = False
            return response
            
        # Get the lane information for the requested color
        lane_key = color_map[req.color]
        lane_info = self.detected_lanes[lane_key]
        
        response.detected = lane_info['detected']
        if lane_info['detected'] and lane_info['bbox'] is not None:
            x, y, w, h = lane_info['bbox']
            response.x = x
            response.y = y
            response.width = w
            response.height = h
            response.area = w * h
            
            # Calculate the bottom center point of the bounding box
            # This is typically where the lane meets the ground
            bottom_center_x = x + w // 2
            bottom_center_y = y + h + self.org_img_h // 4 # last term is to adjust for image resize and cropping
            
            # Project to ground coordinates (multiplying by 2 because initial image is resized to half the size)
            ground_x, ground_y = self.project_pixel_to_ground(bottom_center_x*2, bottom_center_y*2)
            print(ground_x)
            
            if ground_x is not None and ground_y is not None:
                # Distance is the y-coordinate in the ground plane
                # (assuming x-axis points forward from the robot) (from calibration)
                response.distance = ground_x
                rospy.logdebug(f"{lane_key} lane ground coordinates: x={ground_x:.2f}m, y={ground_y:.2f}m")
            else:
                # Fallback to simple approximation
                response.distance = 1000.0 / (h + 0.1)
        
        return response

    def detect_lane(self, image):
        # add your code here
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Yellow lane detection
        yellow_lower = np.array([20, 101, 105], np.uint8)
        yellow_upper = np.array([29, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
        yellow_mask = cv2.dilate(yellow_mask, np.ones((5, 5), "uint8"))
        
        # White lane detection
        # room a
        # white_lower = np.array([115, 24, 221], np.uint8)
        # white_upper = np.array(  [127, 49, 238], np.uint8)

        # room b
        white_lower = np.array([120, 18, 155], np.uint8)
        white_upper = np.array([128, 39, 255], np.uint8)
        
        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
        white_mask = cv2.dilate(white_mask, np.ones((5, 5), "uint8"))
        
        # Create visualization
        annotated_img = image.copy()
        
        # Process yellow lane
        lane_msg = LaneDistance()
        lane_msg.header.stamp = rospy.Time.now()
        lane_msg.yellow_detected = False

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Calculate bottom center point (closest to robot)
                bottom_center_x = x + w // 2
                bottom_center_y = y + h + self.org_img_h // 4 # extra height term to account for resize+crop
                
                # Convert to original image coordinates (if needed due to resize/crop)
                orig_x = bottom_center_x * 2  # Scale back to original size
                orig_y = bottom_center_y * 2  # Adjust based on your preprocessing
                
                # Project to ground coordinates using your homography
                ground_x, ground_y = self.project_pixel_to_ground(orig_x, orig_y)
                
                if ground_x is not None and ground_y is not None:
                    # Detected successfully
                    lane_msg.yellow_detected = True
                    # ground_y is the lateral distance (negative means left of center)
                    lane_msg.yellow_lateral_distance = ground_y
                    # ground_x is the forward distance
                    lane_msg.yellow_forward_distance = ground_x
                    lane_msg.yellow_x = x + w//2
                    lane_msg.yellow_y = x + h//2
                    
                    # Draw on visualization
                    annotated_img = cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    text = f"Dist: {ground_y:.2f}m"
                    cv2.putText(annotated_img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Process white lane (similar to yellow lane)
        lane_msg.white_detected = False
        
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 200:
                x, y, w, h = cv2.boundingRect(largest_contour)
                bottom_center_x = x + w // 2
                bottom_center_y = y + h + self.org_img_h // 4 
                
                orig_x = bottom_center_x * 2
                orig_y = bottom_center_y * 2
                
                ground_x, ground_y = self.project_pixel_to_ground(orig_x, orig_y)
                
                if ground_x is not None and ground_y is not None:
                    lane_msg.white_detected = True
                    lane_msg.white_lateral_distance = ground_y
                    lane_msg.white_forward_distance = ground_x
                    lane_msg.white_x = x + w//2
                    lane_msg.white_y = x + h//2
                    annotated_img = cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    text = f"Dist:{ground_y:.2f}m"
                    cv2.putText(annotated_img, text, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Publish white lane distance
        self.lane_pub.publish(lane_msg)
        
        return annotated_img
    
    def callback(self, img_msg):
        # add your code here
        if not self.got_camera_info:
            return
        
        # convert compressed image to CV2
        image = self.bridge.compressed_imgmsg_to_cv2(img_msg)

        # undistort image
        undistorted_image = self.undistort_image(image)

        # preprocess image
        preprocessed_image = self.preprocess_image(undistorted_image)
        
        # detect lanes - 2.1 (lane detection results published inside detect_lane())
        lane_detected_image = self.detect_lane(preprocessed_image)
        
        # detect lanes and colors - 1.3
        color_detected_image = self.detect_lane_color(lane_detected_image)
        color_detection_msg = self.bridge.cv2_to_compressed_imgmsg(color_detected_image)
        self.color_detection_pub.publish(color_detection_msg)
        
        # publish undistorted image
        undistorted_msg = self.bridge.cv2_to_compressed_imgmsg(undistorted_image)
        self.undistorted_pub.publish(undistorted_msg)

if __name__ == '__main__':
    node = LaneDetectionNode(node_name='lane_detection_node')
    rospy.spin()