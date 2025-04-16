#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber 
rosrun computer_vision lane_following_template.py --p 45.0 --n p --t 20
 
# wait for app to end
dt-launchfile-join