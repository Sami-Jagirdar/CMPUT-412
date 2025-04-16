#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun computer_vision navigate_template.py --lane blue
 
# wait for app to end
dt-launchfile-join
