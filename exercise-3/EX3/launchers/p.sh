#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber 
rosrun computer_vision lane_following_template.py --p 30.0 --d 0.7 --n pd --t 60
 
# wait for app to end
dt-launchfile-join