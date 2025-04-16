#!/usr/bin/env python3

#1.4 - 1.6

# import required libraries
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, LEDPattern
from lane_detection_template import LaneDetectionNode
from std_msgs.msg import ColorRGBA
import math
from computer_vision.srv import GetLaneInfo
import argparse

VELOCITY = 0.3
OMEGA = 0.0

RED=1
GREEN=2
BLUE=3

class NavigationControl(DTROS):
    def __init__(self, node_name):
        super(NavigationControl, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        # add your code here

        # publisher for wheel commands
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self._vel_publisher = rospy.Publisher(self._twist_topic, Twist2DStamped, queue_size=1)
        # robot params
        self.v = VELOCITY
        self.omega = OMEGA

        # LED
        self.LEDspattern = LEDPattern()
        self.light_color_list = [ # init led lights
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                 ]
        
        self.pub_leds = rospy.Publisher(f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=10)

        # Service clients for lane detection
        self.lane_info_service = self.setup_service_client(f'/{self._vehicle_name}/lane_detection_node/get_lane_info')

        # define other variables as needed
        self.rate = rospy.Rate(10)

    def setup_service_client(self, service_name):
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            return rospy.ServiceProxy(service_name, GetLaneInfo)
        except rospy.ROSException:
            rospy.logwarn(f"Service {service_name} not available, proceeding anyway")
            return None
    
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

    def publish_velocity(self):
        message = Twist2DStamped(v=self.v, omega=self.omega)
        self._vel_publisher.publish(message)
        
    def stop(self):
        rospy.loginfo(f"Stopping.")
        # Publish LED color (Red for stop)
        self.set_led_color([[1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],])
        self.v=0
        self.omega=0
        self.publish_velocity()

    def wait(self, duration):
        rospy.loginfo(f"Stopping for {duration} seconds.")

        # Stop movement
        self.v=0
        self.omega=0
        self.publish_velocity()

        # Wait for the specified duration
        rospy.sleep(duration)
        
    def move_straight(self, distance, speed=0.3):
        rospy.loginfo(f"Moving straight for {distance} m")
    
        # Publish LED color (White for moving forward)
        self.set_led_color([[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],])

        # Calculate duration based on speed (constant velocity assumed)
        duration = distance / speed
        start_time = rospy.Time.now().to_sec()

        while rospy.Time.now().to_sec() - start_time < duration:
            self.v = speed
            self.omega = 0
            self.publish_velocity()
            self.rate.sleep()

        # Stop after moving
        self.stop()
        
    def turn_right(self, angle, radius):
        rospy.loginfo("Turning right 90 degrees.")

        # Publish LED color (Yellow (front right and back right) for right turn)
        self.set_led_color([[1, 1, 1, 1],
                                [1, 1, 0, 1],
                                [1, 1, 1, 1],
                                [1, 1, 0, 1],
                                [1, 1, 1, 1],])

        self.turn(angle, radius, lin_v=-0.4)
        
    def turn_left(self, angle, radius):
        rospy.loginfo("Turning left 90 degrees.")

        # Publish LED color (Yellow (front right and back right) for right turn)
        self.set_led_color([[1, 1, 0, 1],
                                [1, 1, 1, 1],
                                [1, 1, 0, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],])

        self.turn(angle, radius, lin_v=0.5)
        

    # add other functions as needed
    def turn(self, angle, radius, lin_v = 0.6):

        # linear speed and duration
        self.v = abs(lin_v)
        if lin_v < 0:
            self.omega = (lin_v-0.7)/radius
        else:
            self.omega = (lin_v+0.7)/radius
        duration = angle/abs(self.omega)
        # duration = 5
        start_time = rospy.Time.now().to_sec()

        while rospy.Time.now().to_sec() - start_time < duration+0.7:
            self.publish_velocity()
            self.rate.sleep()

        # Stop after turning
        rospy.loginfo(f"duration: {rospy.Time.now().to_sec() - start_time}")
        self.stop()

    def execute_red_lane_behaviour(self):
        rospy.loginfo("Executing Red Lane Behaviour")

        self.set_led_color([[2, 1, 0, 1],
                            [2, 1, 0, 1],
                            [2, 1, 0, 1],
                            [2, 1, 0, 1],
                            [2, 1, 0, 1],
                            ])
        
        detected_red_line = False
        stop_distance = 0.20

        self.v = 0.2
        self.omega = 0

        start_time = rospy.Time.now().to_sec()
        timeout = 15

        # Approach red lane and stop before it
        while not detected_red_line and rospy.Time.now().to_sec() - start_time < timeout:
            if self.lane_info_service:
                try:
                    response = self.lane_info_service(RED)
                    if response.detected:
                        distance_to_line = response.distance
                        rospy.loginfo(f"Red line detected: w x h = {response.width} x {response.height}, distance = {distance_to_line:.2f}m \n")

                        if distance_to_line <= stop_distance:
                            rospy.loginfo(f"reached stop distance")
                            detected_red_line = True
                            break
                    
                    self.publish_velocity()
                    self.rate.sleep()
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service call failed: {e}")
                    break
            else:
                rospy.logwarn("Red Lane service not availble")
                breakself.set_led_color([[2, 1, 0, 1],
                            [2, 1, 0, 1],
                            [2, 1, 0, 1],
                            [2, 1, 0, 1],
                            [2, 1, 0, 1],
                            ])
        
        self.set_led_color([[1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            ])
        rospy.loginfo(f"Stopping before red lane for 4 seconds")
        self.wait(4) # Wait for 4 seconds
        
        rospy.loginfo(f"Continuing past the red lane for 1.25m...")
        self.move_straight(1.25)

        rospy.loginfo(f"Completed Red Line behaviour")

    def execute_green_lane_behaviour(self):
        rospy.loginfo("Executing GREEN Lane Behaviour")

        self.set_led_color([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            ])
        
        detected_green_line = False
        stop_distance = 0.20

        self.v = 0.3
        self.omega = 0

        start_time = rospy.Time.now().to_sec()
        timeout = 15

        # Approach red lane and stop before it
        while not detected_green_line and rospy.Time.now().to_sec() - start_time < timeout:
            if self.lane_info_service:
                try:
                    response = self.lane_info_service(GREEN)
                    if response.detected:
                        distance_to_line = response.distance
                        rospy.loginfo(f"GREEN line detected: w x h = {response.width} x {response.height}, distance = {distance_to_line:.2f}m \n")

                        if distance_to_line <= stop_distance:
                            rospy.loginfo(f"reached stop distance")
                            detected_green_line = True
                            break
                    
                    self.publish_velocity()
                    self.rate.sleep()
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service call failed: {e}")
                    break
            else:
                rospy.logwarn("GREEN Lane service not availble")
                break

        self.set_led_color([[0, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1],
                            ])
        
        rospy.loginfo(f"Stopping before GREEN lane for 4 seconds")
        self.wait(4) # Wait for 4 seconds
        
        rospy.loginfo(f"Turning LEFT past GREEN lane")
        self.turn_left(math.pi, 0.3)

        rospy.loginfo(f"Completed GREEN Line behaviour")

    def execute_blue_lane_behaviour(self):
        rospy.loginfo("Executing BLUE Lane Behaviour")

        self.set_led_color([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            ])
        
        detected_blue_line = False
        stop_distance = 0.20

        self.v = 0.4
        self.omega = 0

        start_time = rospy.Time.now().to_sec()
        timeout = 15

        # Approach red lane and stop before it
        while not detected_blue_line and rospy.Time.now().to_sec() - start_time < timeout:
            if self.lane_info_service:
                try:
                    response = self.lane_info_service(BLUE)
                    if response.detected:
                        distance_to_line = response.distance
                        rospy.loginfo(f"BLUE line detected: w x h = {response.width} x {response.height}, distance = {distance_to_line:.2f}m \n")

                        if distance_to_line <= stop_distance:
                            rospy.loginfo(f"reached stop distance")
                            detected_blue_line = True
                            break
                    
                    self.publish_velocity()
                    self.rate.sleep()
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service call failed: {e}")
                    break
            else:
                rospy.logwarn("BLUE Lane service not availble")
                break
            
        self.set_led_color([[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1],
                            ])
        rospy.loginfo(f"Stopping before BLUE lane for 4 seconds")
        self.wait(4) # Wait for 4 seconds
        
        rospy.loginfo(f"Turning RIGHT past BLUE lane")
        self.turn_right(math.pi, 0.2)

        rospy.loginfo(f"Completed BLUE Line behaviour")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane-following behavior controller')
    parser.add_argument('--lane', type=str, choices=['red', 'green', 'blue', 'none'], 
                      default='none', help='Lane behavior to execute (red, green, blue, or none)')
    
    # Parse arguments provided by ROS launch filerospy.Time.now().to_sec() - start_time
    # Note: We need to parse the arguments from the ROS command line or add these in the launchers
    args = parser.parse_args(rospy.myargv()[1:])
 
    node = NavigationControl(node_name='navigation_control_node')
    
    # Wait a bit for the service to be available
    rospy.sleep(2)
    
    if args.lane == 'red':
        node.execute_red_lane_behaviour()
    elif args.lane == 'green':
        node.execute_green_lane_behaviour()
    elif args.lane == 'blue':
        node.execute_blue_lane_behaviour()
    else:
        rospy.loginfo("No lane behavior specified, node is ready and waiting for commands.")

    rospy.signal_shutdown("complete")
    # Keep the node running
    rospy.spin()