# visual_mapping
semantic mapping based on visual perception


01312021
Integrate PSPNet into ROS.
PSPNet use Python3 from Conda
MaskRCNN use Python 2.7 from ROS
So, with optional sys.path.insert and remove, we are able to load two model into one ros workspace.
And this is technique to support both Python2.7 and Python3.
This method is very useful for the future develop. Even though when we have different version of Python3.

