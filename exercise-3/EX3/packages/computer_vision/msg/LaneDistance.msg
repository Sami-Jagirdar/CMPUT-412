# LaneDistance.msg
Header header
# Yellow lane information
bool yellow_detected
float32 yellow_lateral_distance  # meters, positive means left of vehicle center
float32 yellow_forward_distance  # meters, forward from vehicle
float32 yellow_x  # meters, positive means left of vehicle center
float32 yellow_y  # meters, forward from vehicle


# White lane information
bool white_detected
float32 white_lateral_distance   # meters, positive means left of vehicle center
float32 white_forward_distance   # meters, forward from vehicle
float32 white_x   # meters, positive means left of vehicle center
float32 white_y   # meters, forward from vehicle