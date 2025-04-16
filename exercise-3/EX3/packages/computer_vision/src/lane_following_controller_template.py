#!/usr/bin/env python3

# potentially useful for question - 2.2

# import required libraries
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from computer_vision.msg import LaneDistance

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        # add your code here
        self._vehicle_name = os.environ['VEHICLE_NAME']
        
        self.controller_type = "pid"  # Can be p, pd or pid
        
        # PID gains 
        self.Kp = 30.0  # Proportional gain
        self.Ki = 0.1   # Integral gain
        self.Kd = 0.7   # Derivative gain
        
        # control variables
        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_error = 0.0
        self.error = 0.0
        
        # movement parameters
        self.max_speed = 0.3      
        self.min_speed = 0.1     
        self.max_omega = 5.0
        self.omega = 5.0
        
        # distance tracking
        self.start_distance = None
        self.current_distance = 0.0
        self.target_distance = 1.5  # Target distance to travel (meters)
        self.is_moving = False
        self.duration = 12 # Set this to make your bot move for the specified time
        
        # initialize publisher/subscribers
        self.lane_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/lane_detection_node/lane_info",
            LaneDistance,
            self.lane_callback
        )
        
        self.cmd_vel_pub = rospy.Publisher(
            f'/{self._vehicle_name}/car_cmd_switch_node/cmd',
            Twist2DStamped,
            queue_size=1
        )
        
        # Variables to store lane information
        self.yellow_lane_detected = False
        self.yellow_lane_lateral_distance = 0.0  # Lateral distance in meters
        self.white_lane_detected = False
        self.white_lane_lateral_distance = 0.0   # Lateral distance in meters
        
        # Target lateral position (ideally in the middle between white and yellow lanes)
        self.target_lateral_position = 0.0  # meters from center
        
        # Time tracking for integral and derivative control
        self.last_callback_time = rospy.get_time()
        self.start_time = rospy.Time.now().to_sec()

        rospy.loginfo(f"Lane controller initialized with {self.controller_type} control")
        rospy.Rate(20)

    def calculate_p_control(self, error):
        # add your code here
        return self.Kp * error


    def calculate_pd_control(self, error, dt):
        # add your code here
        if dt > 0:
            self.derivative = (error - self.prev_error) / dt
        else:
            self.derivative = 0
            
        # Store current error for next iteration
        self.prev_error = error
        
        # Calculate and return control output
        p_term = self.Kp * error
        d_term = self.Kd * self.derivative
        
        return p_term + d_term

    def calculate_pid_control(self, error, dt):
        # add your code here
        # Calculate integral term
        rospy.loginfo(f"dt: {dt}")
        if dt > 0:
            self.integral += error * dt
            # Anti-windup: limit the integral term
            # self.integral = max(min(self.integral, 1.0), -1.0)
            
            # Calculate derivative term
            self.derivative = (error - self.prev_error) / dt
        else:
            self.derivative = 0
            
        # Store current error for next iteration
        self.prev_error = error
        
        # Calculate and return control output
        p_term = self.Kp * error
        i_term = self.Ki * self.integral
        d_term = self.Kd * self.derivative
        
        return p_term + i_term + d_term

    def get_control_output(self, error):
        # add your code here
        current_time = rospy.get_time()
        dt = current_time - self.last_callback_time
        self.last_callback_time = current_time

        if self.controller_type == "p":
            return self.calculate_p_control(error)
        elif self.controller_type == "pd":
            return self.calculate_pd_control(error, dt)
        elif self.controller_type == "pid":
            return self.calculate_pid_control(error, dt)
        else:
            rospy.logwarn(f"Unknown contself.current_distance >= self.target_distanceroller type: {self.controller_type}, using P control")
            return self.calculate_p_control(error)

    def publish_cmd(self, omega, speed=None):
        # If we've reached the target distance, stop
        time = rospy.Time.now().to_sec()
        if time - self.start_time >= self.duration:
            if self.is_moving:
                rospy.loginfo(f"duration ended")
                self.is_moving = False
            
            # Stop the robot
            cmd_msg = Twist2DStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.v = 0.0
            cmd_msg.omega = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
            rospy.signal_shutdown("duration limit reached")
            return
            
        # If we're not yet moving, start tracking distance
        if not self.is_moving:
            self.is_moving = True
            self.start_distance = 0
            rospy.loginfo("Starting lane following...")
        
        # If speed is not specified, use default speed
        if speed is None:
            speed = self.max_speed
        
        # Limit angular velocity
        omega = max(min(omega, self.max_omega), -self.max_omega)
        rospy.loginfo(f"omega: {omega}")
        # Create and publish velocity command
        cmd_msg = Twist2DStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.v = speed
        cmd_msg.omega = omega
        self.cmd_vel_pub.publish(cmd_msg)
        
    def lane_callback(self, msg):
        self.yellow_lane_detected = msg.yellow_detected
        if msg.yellow_detected:
            self.yellow_lane_lateral_distance = msg.yellow_lateral_distance
            self.yellow_lane_forward_distance = msg.yellow_forward_distance

        self.white_lane_detected = msg.white_detected
        if msg.yellow_detected:
            self.white_lane_lateral_distance = msg.white_lateral_distance
            self.white_lane_forward_distance = msg.white_forward_distance

        if msg.yellow_detected or msg.white_detected:
            self.update_control()

        
    def update_control(self):

        if self.yellow_lane_detected and self.white_lane_detected:
            rospy.loginfo(f"yellow and white lane distance: {self.yellow_lane_lateral_distance} , {self.white_lane_lateral_distance}")
            # Note: if yellow lane is to the left of bot, yellow_lateral_distance > 0
            # If white lane is to the right bot, white_lateral_distance < 0
            # The bot itself should be in the center laterally (at y=0)
            target_position = (self.yellow_lane_lateral_distance + self.white_lane_lateral_distance)

            # Error > 0 --> bot is veering to the left of target position(ccw), give -ve omega (cw)
            # Error < 0 --> bot is veering to the right of target position(cw), give +ve omega (ccw)
            self.error = 0.0 + target_position #small error offset because white is irregular so be closer to yellow
            rospy.loginfo(f"error: {self.error}")

            omega = self.get_control_output(self.error)
            speed_factor = 0.6 - min(0.5, abs(self.error))
            forward_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
            self.publish_cmd(omega, forward_speed)

        # Prioritize distance to yellow lane over white lane
        elif self.yellow_lane_detected:
            rospy.loginfo(f"yellow lane distance: {self.yellow_lane_lateral_distance}")
            # Only yellow lane detected, maintain fixed offset
            # Usually yellow lane is on the left, so we want to stay a bit to the right
            # Based on homography from robot's POV, left is +ve y-axis so we add +ve distance
            target_distance = 0.10  # meters

            self.error = self.yellow_lane_lateral_distance - target_distance
            rospy.loginfo(f"error: {self.error}")
            TURN_FACTOR = 1.2
            omega = TURN_FACTOR*self.get_control_output(self.error)
            rospy.loginfo(f"omega: {omega}")
            # Reduce speed when only one lane detected
            speed_factor = 0.6 - min(0.5, abs(self.error))
            forward_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
            
            self.publish_cmd(omega, forward_speed)

        elif self.white_lane_detected:
            rospy.loginfo(f"white lane detected, distance: {self.white_lane_lateral_distance}")
            # Only white lane detected, maintain fixed offset
            # White lane is usually on the right, so we want to stay a bit to the left
            # Based on homography from robot's POV, right is -ve y-axis so we add +ve offset
            target_offset = -0.10  # meters
            # target_lateral = self.white_lane_lateral_distance + target_offset
            
            # self.error = 0.0 + target_lateral
            self.error = self.white_lane_lateral_distance - target_offset
            TURN_FACTOR = 1.2
            omega = TURN_FACTOR*self.get_control_output(self.error)
            
            # Reduce speed when only one lane detected
            speed_factor = 0.6 - min(0.5, abs(self.error))
            forward_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
            
            self.publish_cmd(omega, forward_speed)

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()