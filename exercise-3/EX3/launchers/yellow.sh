#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber 
rosrun computer_vision yellow_lane_following.py --p 30.0 --d 1.0 --n pd --t 300
 
# wait for app to end
dt-launchfile-join