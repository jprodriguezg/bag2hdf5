#!/usr/bin/env python3
"""
This code is inspired in
Code from : https://github.com/uzh-rpg/rpg_ev-transfer/blob/5c953692c6a0e06a3810830fec4ef8db5de0ddb6/datasets/extract_data_tools/export_data_from_rosbag.py#L122
Additional : https://github.com/TimoStoff/event_utils/blob/master/lib/data_formats/rosbag_to_h5.py
"""

from __future__ import print_function, absolute_import
import argparse
import numpy as np

import h5py
from cv_bridge import CvBridge
import rosbag

def get_rosbag_stats(bag, event_topic, image_topic=None):
    num_event_msgs = 0
    num_img_msgs = 0
    topics = bag.get_type_and_topic_info().topics
    for topic_name, topic_info in topics.items():
        if topic_name == event_topic:
            num_event_msgs = topic_info.message_count
            print('Found events topic: {} with {} messages'.format(topic_name, topic_info.message_count))
        if topic_name == image_topic:
            num_img_msgs = topic_info.message_count
            print('Found image topic: {} with {} messages'.format(topic_name, num_img_msgs))
    return num_event_msgs, num_img_msgs

def find_nearest(dataset, start_idx, search_value, search_gap=10000):
    num_events = dataset.shape[0]
    nearest_value_idx = 0

    for event_batch in range((num_events-start_idx) // search_gap):
        start_pos = start_idx+event_batch*search_gap
        end_pos = min(start_idx+(event_batch+1)*search_gap,
                      num_events)
        selected_events = dataset[start_pos:end_pos]

        nearest_idx = np.searchsorted(
            selected_events, search_value, side="left")

        if nearest_idx != search_gap:
            nearest_value_idx = start_idx+event_batch*search_gap+nearest_idx
            break

    return nearest_value_idx


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--topic_events", type=str)
parser.add_argument("--topic_image", type=str)
parser.add_argument("--camera_width", type=int)
parser.add_argument("--camera_height", type=int)
parser.add_argument("--color", action="store_true")

args = parser.parse_args()

print("Loading bag ...")
# Load dataset
bag = rosbag.Bag(args.data_path, "r")
print("Done ...")
# Check dataset
num_event_msgs, num_img_msgs = get_rosbag_stats(bag, args.topic_events, args.topic_image)
# bridge = CvBridge()

# HDF5 file
dataset = h5py.File(args.output_path, "w")
dvs_data = dataset.create_dataset(
    "davis/events",
    shape=(0, 4),
    maxshape=(None, 4),
    dtype="float64")

# img_data = dataset.create_dataset(
#     "davis/image_raw",
#     shape=(0, args.camera_height, args.camera_width, 3),
#     maxshape=(None, args.camera_height, args.camera_width, 3),
#     dtype="uint8")

img_stamp = dataset.create_dataset(
    "davis/image_raw_ts",
    shape=(0, 1),
    maxshape=(None, 1),
    dtype="float64")

img_ind = dataset.create_dataset(
    "davis/image_raw_event_inds",
    shape=(0,),
    maxshape=(None,),
    dtype="int64")

topics = [args.topic_image, args.topic_events]

current_frame_ts = None

event_ts_collector = np.zeros((0,), dtype="float64")
frame_ts_collector = np.zeros((0,), dtype="float64")

print("Reading bag .... ")
event_msg = 0
image_msg = 0
for topic, msg, t in bag.read_messages(topics=topics):

    if topic == topics[1]:
        events = msg.events
        num_events = len(events)

        # save events
        dvs_data.resize(dvs_data.shape[0]+num_events, axis=0)

        event_data = np.array(
            [[e.x, e.y, float(e.ts.to_nsec())/1e9,
              e.polarity] for e in events],
            dtype="float64")

        dvs_data[-num_events:] = event_data

        event_ts_collector = np.append(
            event_ts_collector, [float(x.ts.to_nsec())/1e9 for x in events])
        #print("Processed {} events".format(num_events))
        print("Event msg %d of %d"%(event_msg,num_event_msgs))
        event_msg += 1

    elif topic in topics[0]:
        # im = bridge.imgmsg_to_cv2(msg, "rgb8")
        try:
            # save image
            # img_data.resize(img_data.shape[0]+1, axis=0)
            # img_data[-1] = im
            current_frame_ts = float(msg.header.stamp.to_nsec())/1e9
            frame_ts_collector = np.append(
                frame_ts_collector, [current_frame_ts], axis=0)

            img_stamp.resize(img_stamp.shape[0]+1, axis=0)
            img_stamp[-1] = current_frame_ts
            #print("Processed frame.")
        except TypeError:
            print("Some error")
            continue
bag.close()
print("Bag reading done")

print("Fixing event index ... ")
# search for nearest event index
nearest_idx = 0
for frame_ts in frame_ts_collector:
    nearest_idx = find_nearest(event_ts_collector, nearest_idx, frame_ts)

    img_ind.resize(img_ind.shape[0]+1, axis=0)
    img_ind[-1] = nearest_idx

print("Done ... ")
dataset.close()
