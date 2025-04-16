#!/usr/bin/env python3

# Part 3

# import required libraries
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from computer_vision.msg import LaneDistance
import argparse
import math

class LaneControllerNode(DTROS):
    def __init__(self, node_name, kp=30.0, ki=0.05, kd=2.2, controller_type="pd", duration=15, drive=0):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        # add your code here
        self._vehicle_name = os.environ['VEHICLE_NAME']
        
        if controller_type not in ["p", "pd", "pid"]:
            rospy.logwarn(f"Unknown controller type: {controller_type}, using pd control")
            self.controller_type = "pd"
        else:
            self.controller_type = controller_type  # Can be p, pd or pid
        
        # PID gains 
        self.Kp = kp  # Proportional gain
        self.Ki = ki   # Integral gain
        self.Kd = kd   # Derivative gain
        rospy.loginfo(f"Using {self.controller_type} controller with Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")
        
        # control variables
        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_error = 0.0
        self.error = 0.0
        
        # movement parameters
        self.max_speed = 0.2      
        self.min_speed = 0.1     
        self.max_omega = 5.0
        
        # time tracking
        self.is_moving = False
        self.duration = duration # Set this to make your bot move for the specified time

        self.drive = drive
        
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
        self.cmd_msg = Twist2DStamped()
        
        # Variables to store lane information
        self.yellow_lane_detected = False
        self.yellow_lane_lateral_distance = 0.0  # Lateral distance in meters
        self.white_lane_detected = False
        self.white_lane_lateral_distance = 0.0   # Lateral distance in meters
        
        # Time tracking for integral and derivative control
        self.last_callback_time = rospy.get_time()
        self.start_time = rospy.Time.now().to_sec()

        rospy.Rate(20)

        rospy.on_shutdown(self.shutdown_hook)

    def shutdown_hook(self):
        # Publish zero velocity command when shutting down
        cmd_msg = Twist2DStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.cmd_vel_pub.publish(cmd_msg)
        rospy.loginfo("Robot stopped due to shutdown")

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
        rospy.loginfo(f"dt: {dt:0.3f}")
        if dt > 0:
            self.integral += error * dt
            # Anti-windup: limit the integral term
            self.integral = max(min(self.integral, 1.0), -1.0)
            if abs(self.integral == 1):
                print(self.integral)
            
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
        # If we've reached the target duration, stop
        time = rospy.Time.now().to_sec()
        if time - self.start_time >= self.duration:
            if self.is_moving:
                rospy.loginfo(f"duration ended")
                self.is_moving = False
            
            # Stop the robot
            # cmd_msg = Twist2DStamped()
            self.cmd_msg.header.stamp = rospy.Time.now()
            self.cmd_msg.v = 0.0
            self.cmd_msg.omega = 0.0
            self.cmd_vel_pub.publish(self.cmd_msg)
            rospy.signal_shutdown("duration limit reached")
            return
            
        # If we're not yet moving, start moving
        if not self.is_moving:
            self.is_moving = True
            rospy.loginfo("Starting lane following...")
        
        # If speed is not specified, use default speed
        if speed is None:
            speed = self.max_speed
        
        # Limit angular velocity
        omega = max(min(omega, self.max_omega), -self.max_omega)
        rospy.loginfo(f"omega: {omega:.3f}\n")
        # Create and publish velocity command
        self.cmd_msg.header.stamp = rospy.Time.now()
        self.cmd_msg.v = speed
        self.cmd_msg.omega = omega
        self.cmd_vel_pub.publish(self.cmd_msg)
        
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

        # Prioritize distance to yellow lane over white lane
        if self.yellow_lane_detected:
            rospy.loginfo(f"yellow lane distance: {self.yellow_lane_lateral_distance}")

            if self.drive == 0:
                target_distance = 0.08  # meters
            else:
                target_distance = -0.08

            self.error = self.yellow_lane_lateral_distance - target_distance
            rospy.loginfo(f"error: {self.error}")
            omega = self.get_control_output(self.error)
            # Reduce speed when only one lane detected
            speed_factor = 0.6 - min(0.5, abs(self.error))
            forward_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
            
            self.publish_cmd(omega, forward_speed)

        elif self.white_lane_detected:
            rospy.loginfo(f"white lane distance: {self.white_lane_lateral_distance:0.3f}, {self.white_lane_forward_distance:0.3f}")

            if self.drive == 0:
                target_distance = -0.15  # meters
            else:
                target_distance = 0.07

            self.error = self.white_lane_lateral_distance - target_distance
            if self.white_lane_forward_distance > 0.16:
                self.error += 0.05
            rospy.loginfo(f"error: {self.error}")
            omega = self.get_control_output(self.error)
            # Reduce speed when only one lane detected
            speed_factor = 0.6 - min(0.5, abs(self.error))
            forward_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
            
            self.publish_cmd(omega, 0.1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='lane-following')
    
    # Add pid gain args
    parser.add_argument('--p', type=float, 
                      default='30.0', help='Proportional gain')
    parser.add_argument('--i', type=float, 
                      default='0.1', help='Integral gain')
    parser.add_argument('--d', type=float, 
                      default='1.0', help='Derivative gain')
    parser.add_argument('--n', type=str,
                        default='pd', help='Controller type (p, pd, pid)')
    parser.add_argument('--t', type=float, default='15', help='duration')
    parser.add_argument('--drive', type=int, default='0', help='english or american drive')
    # english = 1, american = 0
    
    args = parser.parse_args(rospy.myargv()[1:])

    node = LaneControllerNode(node_name='lane_controller_node', kp=args.p, ki=args.i, kd=args.d, controller_type=args.n, duration=args.t, drive=args.drive)
    
    rospy.spin()