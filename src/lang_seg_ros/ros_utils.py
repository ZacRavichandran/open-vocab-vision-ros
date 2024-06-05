import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Header

from lang_seg_ros.msg import Track
from lang_seg_ros.tracker import Hypothesis


def header_from_track(track: Hypothesis) -> Header:
    header = Header()
    header.frame_id = track.frame
    header.stamp.secs = np.int32(track.time // 1)
    header.stamp.nsecs = np.int32(track.time % 1 * 1e9 // 1)
    return header


def to_track_msg(track: Hypothesis) -> Track:
    track_msg = Track()
    track_msg.header = header_from_track(track)
    track_msg.idx = track.idx
    track_msg.label = track.label
    track_msg.class_id = track.class_id
    track_msg.pose.pose.position = Point(
        x=track.pose[0], y=track.pose[1], z=track.pose[2]
    )
    track_msg.pose.pose.orientation.w = 1
    return track_msg


def from_track_msg(track_msg: Track) -> Hypothesis:
    pose = np.array(
        [
            track_msg.pose.pose.position.x,
            track_msg.pose.pose.position.y,
            track_msg.pose.pose.position.z,
        ]
    )
    det_time = track_msg.header.stamp.secs + track_msg.header.stamp.nsecs / 1e9

    return Hypothesis(
        class_id=track_msg.class_id,
        idx=track_msg.idx,
        pose=pose,
        frame=track_msg.header.frame_id,
        time=det_time,
        score=1,
        label=track_msg.label,
    )
