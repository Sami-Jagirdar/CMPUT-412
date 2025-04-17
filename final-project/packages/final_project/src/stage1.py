#!/usr/bin/env python3

import rospy
import os
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point32
import cv2
from general_navigation import NavigationControl
from cv_bridge import CvBridge
import numpy as np


class TailDuckNode(DTROS):
    def __init__(self, node_name):
        super(TailDuckNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']

        self.last_stamp = rospy.Time.now()
        self.process_frequency = 3
        self.circlepattern_dims = [7, 3]
        self.blobdetector_min_area = 10
        self.blobdetector_min_dist_between_blobs = 2
        self.cbParametersChanged() 

        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.image_callback,
                                    queue_size=1,
                                    buff_size="20MB")

        # self.pub_centers = rospy.Publisher("/{}/duckiebot_detection_node/centers".format(os.environ['VEHICLE_NAME']), VehicleCorners, queue_size=1)
        self.pub_circlepattern_image = rospy.Publisher("/{}/duckiebot_detection_node/detection_image/compressed".format(os.environ['VEHICLE_NAME']), CompressedImage, queue_size=1)
        # self.pub_detection = rospy.Publisher("/{}/duckiebot_detection_node/detection".format(os.environ['VEHICLE_NAME']), BoolStamped, queue_size=1)
        self.log("Detection Initialization completed.")

        self.nav = NavigationControl()
        self.velocity = 0
        self.omega = 0
        self.nav.publish_velocity(self.velocity, self.omega)

        rospy.on_shutdown(self.hook)


    def cbParametersChanged(self):
        self.publish_duration = rospy.Duration.from_sec(1.0 / self.process_frequency)
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = self.blobdetector_min_area
        params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)


    def detect_bot(self, image_cv):
        """
        Callback for processing a image which potentially contains a back pattern. Processes the image only if
        sufficient time has passed since processing the previous image (relative to the chosen processing frequency).

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

            if self.pub_circlepattern_image.get_num_connections() > 0:
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

        now = rospy.Time.now()
        if now - self.last_stamp < self.publish_duration:
            return
        else:
            self.last_stamp = now

        image_cv = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        result = self.detect_bot(image_cv)

        if result is not None:
            error_distance, offset = result

            if error_distance < 10:
                self.nav.publish_velocity(0,0)
                return

            # Tuning parameters
            Kp_dist = 0.01
            Kp_angle = -0.05

            # Compute velocity and omega based on error
            v = Kp_dist * error_distance
            omega = Kp_angle * offset

            # Limit speed to avoid overshooting
            v = max(min(v, 0.35), 0.05) if v > 0 else 0

            rospy.loginfo(f"[Tailing] error={error_distance:.1f}, offset={offset:.1f} => v={v:.2f}, omega={omega:.2f}")
            self.nav.publish_velocity(v, omega)

        else:
            # Stop if bot not visible
            self.nav.publish_velocity(0, 0)

        # vel = 0.3
        # twist = 0

        # if bot_detected or self.manuvering:
        #     self.manuvering = True
        #     vel, twist = self.manuver_around_bot()

        # # print(vel, twist)
        # self.nav.publish_velocity(vel, twist)

    def hook(self):
        print("SHUTTING DOWN")
        for i in range(8):
            self.nav.publish_velocity(0,0)


if __name__ == '__main__':
    # create the node
    node = TailDuckNode(node_name='tail_duck_node')
    rospy.spin()