# Duckiebot Exercise 3 - Code Implementation

## Collaborators
Sami Jagirdar [ccid: jagirdar, sid: 1686267]
Basia Ofovwe [ccid: ofovwe, sid: 1667223]

# Running Lane detection (1.1-1.3 + 2.1)
code is in \packages\computer_vision\src\lane_detection_template.py
run using launcher lane-detection.sh

# Running Navigation and LED control (1.4-1.5)
code is in \packages\computer_vision\src\navigate.py
red line behaviour launcher red-line.sh
blue line behaviour launcher blue-line.sh
green line behaviour launcher green-line.sh

# Running PID controllers and Lane Following (2.2 and 3)
code is in \packages\computer_vision\src\lane_following_controller_template.py
run using lane-controller.sh

Alternatively to directly give pid gain values from command line,
use \packages\computer_vision\src\alt_lane_following_controller_template.py
Copy the computer_vision package to the docker container running dt-gui-tools in the packages folder
Run catkin build
then run source devel/build


# Template: template-ros

This template provides a boilerplate repository
for developing ROS-based software in Duckietown.

**NOTE:** If you want to develop software that does not use
ROS, check out [this template](https://github.com/duckietown/template-basic).


## How to use it

### 1. Fork this repository

Use the fork button in the top-right corner of the github page to fork this template repository.


### 2. Create a new repository

Create a new repository on github.com while
specifying the newly forked template repository as
a template for your new repository.


### 3. Define dependencies

List the dependencies in the files `dependencies-apt.txt` and
`dependencies-py3.txt` (apt packages and pip packages respectively).


### 4. Place your code

Place your code in the directory `/packages/` of
your new repository.


### 5. Setup launchers

The directory `/launchers` can contain as many launchers (launching scripts)
as you want. A default launcher called `default.sh` must always be present.

If you create an executable script (i.e., a file with a valid shebang statement)
a launcher will be created for it. For example, the script file 
`/launchers/my-launcher.sh` will be available inside the Docker image as the binary
`dt-launcher-my-launcher`.

When launching a new container, you can simply provide `dt-launcher-my-launcher` as
command.

