"""
Post-process a rosbag containing /image_raw and /movement_control topics
into a structured dataset for training steering models.
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions


def get_rosbag_messages(bag_path: str, topic: str) -> List[Tuple[float, any]]:
    """
    Extract all messages from a specific topic in a rosbag.

    Returns:
        List of (timestamp, message) tuples
    """
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic type
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if topic not in type_map:
        print(f"Warning: Topic '{topic}' not found in bag")
        return []

    msg_type = get_message(type_map[topic])
    messages = []

    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        if topic_name == topic:
            msg = deserialize_message(data, msg_type)
            timestamp_sec = timestamp / 1e9  # Convert nanoseconds to seconds
            messages.append((timestamp_sec, msg))

    return messages


def synchronize_messages(
    image_msgs: List[Tuple[float, any]], control_msgs: List[Tuple[float, any]], max_time_diff: float = 0.1
) -> List[Tuple[float, any, any]]:
    """
    Synchronize image and control messages by timestamp.

    For each image, find the closest control message within max_time_diff.

    Returns:
        List of (timestamp, image_msg, control_msg) tuples
    """
    synchronized = []
    control_idx = 0

    for img_time, img_msg in image_msgs:
        # Find closest control message
        best_control = None
        best_diff = float("inf")

        # Search forward from last position
        search_idx = control_idx
        while search_idx < len(control_msgs):
            ctrl_time, ctrl_msg = control_msgs[search_idx]
            time_diff = abs(ctrl_time - img_time)

            if time_diff < best_diff:
                best_diff = time_diff
                best_control = ctrl_msg
                control_idx = search_idx

            # Stop searching if we're moving away in time
            if ctrl_time > img_time + max_time_diff:
                break

            search_idx += 1

        if best_control is not None and best_diff <= max_time_diff:
            synchronized.append((img_time, img_msg, best_control))
        else:
            print(f"Warning: No control message found within {max_time_diff}s for image at {img_time:.3f}")

    return synchronized


def process_rosbag(
    bag_path: str,
    output_dir: str,
    image_topic: str = "/image_raw",
    control_topic: str = "/movement_control",
    max_sync_diff: float = 0.1,
) -> None:
    """
    Process a rosbag and create a dataset directory.

    Args:
        bag_path: Path to the rosbag directory
        output_dir: Output dataset directory
        image_topic: Topic name for images
        control_topic: Topic name for control commands
        max_sync_diff: Max time difference for synchronizing messages (seconds)
    """
    print(f"Processing rosbag: {bag_path}")
    print(f"Output directory: {output_dir}")

    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Extract messages
    print(f"\nExtracting messages from {image_topic}...")
    image_messages = get_rosbag_messages(bag_path, image_topic)
    print(f"Found {len(image_messages)} image messages")

    print(f"\nExtracting messages from {control_topic}...")
    control_messages = get_rosbag_messages(bag_path, control_topic)
    print(f"Found {len(control_messages)} control messages")

    if not image_messages:
        print(f"Error: No messages found on {image_topic}")
        return

    if not control_messages:
        print(f"Error: No messages found on {control_topic}")
        return

    # Synchronize messages
    print(f"\nSynchronizing messages (max time diff: {max_sync_diff}s)...")
    synchronized = synchronize_messages(image_messages, control_messages, max_sync_diff)
    print(f"Successfully synchronized {len(synchronized)} image-control pairs")

    if not synchronized:
        print("Error: No synchronized messages found")
        return

    # Process and save data
    bridge = CvBridge()
    labels_path = output_path / "labels.csv"

    print(f"\nWriting dataset to {output_dir}...")
    with open(labels_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "timestamp", "steering"])

        for idx, (timestamp, img_msg, ctrl_msg) in enumerate(synchronized, start=1):
            # Save image
            image_filename = f"{idx:06d}.png"
            image_path = images_dir / image_filename

            try:
                cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                cv2.imwrite(str(image_path), cv_image)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue

            # Write label
            steering_rad = math.radians(float(ctrl_msg.steering_angle))
            writer.writerow([idx, f"{timestamp:.6f}", f"{steering_rad:.6f}"])

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(synchronized)} samples...")

    print(f"\n✓ Dataset created successfully!")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_path}")
    print(f"  Total samples: {len(synchronized)}")


def main():
    parser = argparse.ArgumentParser(description="Process rosbag into structured dataset for steering model training")
    parser.add_argument("bag_path", type=str, help="Path to the rosbag directory")
    parser.add_argument("output_dir", type=str, help="Output dataset directory")
    parser.add_argument("--image-topic", type=str, default="/image_raw", help="Image topic name (default: /image_raw)")
    parser.add_argument(
        "--control-topic", type=str, default="/movement_control", help="Control topic name (default: /movement_control)"
    )
    parser.add_argument(
        "--max-sync-diff",
        type=float,
        default=0.1,
        help="Maximum time difference for message synchronization in seconds (default: 0.1)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.bag_path):
        print(f"Error: Rosbag path '{args.bag_path}' does not exist")
        return 1

    try:
        process_rosbag(args.bag_path, args.output_dir, args.image_topic, args.control_topic, args.max_sync_diff)
        return 0
    except Exception as e:
        print(f"Error processing rosbag: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
