from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion

from typing import Sequence


IDENTITY_QUATERNION = Quaternion(x=0, y=0, z=0, w=1)
BASE_MARKER = Marker()


def create_marker_msg(
    *,
    id: int,
    header: Header,
    position: Sequence[float],
    color: ColorRGBA,
    orientation: Quaternion = IDENTITY_QUATERNION,
    scale: float = 0.25,
    marker_type=BASE_MARKER.SPHERE
) -> Marker:
    marker_msg = Marker()
    marker_msg.id = id
    marker_msg.header = header
    marker_msg.pose.position = Point(x=position[0], y=position[1], z=position[2])
    marker_msg.pose.orientation = orientation

    marker_msg.color = color
    marker_msg.scale.x = scale
    marker_msg.scale.y = scale
    marker_msg.scale.z = scale
    marker_msg.action = marker_msg.ADD
    marker_msg.type = marker_type

    return marker_msg
