#!/usr/bin/env python3
"""
bag_id_shift.py — clone a rosbag2 while offsetting Marker IDs per "run"

Usage:
  python3 bag_id_shift.py \
    --in /path/to/input_bag_dir \
    --out /path/to/output_bag_dir \
    --gap-sec 1.0 \
    --id-offset 100000 \
    --append-ns

What it does:
- Iterates messages chronologically from the input bag.
- Increments run_id when it detects a large time gap (gap-sec) OR a Marker with action==DELETEALL.
- For visualization_msgs/Marker & MarkerArray:
    * id := id + run_id * id_offset
    * (optional) ns := f"run{run_id}/{ns}"
- Writes everything (modified or not) to a new bag with original timestamps.
"""

import argparse
import sys
import os
import math
from typing import Dict

# ROS 2 bag + (de)serialization
try:
    import rosbag2_py
except Exception as e:
    print("ERROR: rosbag2_py not importable. Run inside a ROS 2 environment.", file=sys.stderr)
    raise

from rclpy.serialization import serialize_message, deserialize_message
from rosidl_runtime_py.utilities import get_message

# --------- Helpers ---------
def normalize_ros_type(type_str: str) -> str:
    # Accept both "pkg/msg/Type" and "pkg/Type"
    if "/msg/" in type_str:
        return type_str
    parts = type_str.split("/")
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return type_str

def is_marker_type(type_str: str) -> bool:
    t = normalize_ros_type(type_str)
    return t in ("visualization_msgs/msg/Marker", "visualization_msgs/msg/MarkerArray")

def shift_marker_obj(m, run_id: int, id_offset: int, append_ns: bool):
    # visualization_msgs/Marker fields: ns (string), id (int32), action (uint8)
    try:
        m.id = int(m.id) + run_id * id_offset
    except Exception:
        pass
    if append_ns:
        try:
            base = m.ns if (hasattr(m, "ns") and m.ns) else ""
            m.ns = (f"run{run_id}/{base}") if base else f"run{run_id}"
        except Exception:
            pass

def shift_markers(msg, type_str: str, run_id: int, id_offset: int, append_ns: bool):
    t = normalize_ros_type(type_str)
    if t == "visualization_msgs/msg/Marker":
        shift_marker_obj(msg, run_id, id_offset, append_ns)
    elif t == "visualization_msgs/msg/MarkerArray":
        if hasattr(msg, "markers"):
            for mk in msg.markers:
                shift_marker_obj(mk, run_id, id_offset, append_ns)

def marker_is_delete_all(msg, type_str: str) -> bool:
    t = normalize_ros_type(type_str)
    try:
        if t == "visualization_msgs/msg/Marker":
            # Marker.DELETEALL constant is 3
            return int(msg.action) == 3
        elif t == "visualization_msgs/msg/MarkerArray":
            # If any element requests DELETEALL treat as boundary
            for mk in getattr(msg, "markers", []):
                if int(mk.action) == 3:
                    return True
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser(description="Clone a rosbag2 with Marker IDs offset per detected run.")
    ap.add_argument("--in", dest="in_uri", required=True, help="Input rosbag2 directory (uri).")
    ap.add_argument("--out", dest="out_uri", required=True, help="Output rosbag2 directory (uri). Must not exist.")
    ap.add_argument("--storage", default="sqlite3", choices=["sqlite3", "mcap"],
                    help="rosbag2 storage plugin to use for reading/writing (default: sqlite3).")
    ap.add_argument("--gap-sec", type=float, default=1.0,
                    help="Run boundary: if wall gap between consecutive messages exceeds this (sec). Default 1.0")
    ap.add_argument("--id-offset", type=int, default=100000,
                    help="ID offset per run: new_id = old_id + run_id * id_offset. Default 100000.")
    ap.add_argument("--append-ns", action="store_true",
                    help="Also prefix Marker.ns with 'run{run_id}/' to avoid namespace collisions.")
    ap.add_argument("--no-deleteall-boundary", action="store_true",
                    help="Disable treating Marker(DELETEALL) as a run boundary.")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_uri)
    if os.path.isdir(out_dir):
        print(f"ERROR: Output path exists: {out_dir}", file=sys.stderr)
        sys.exit(2)

    # Ensure parent exists, but do NOT create out_dir itself;
    # rosbag2 will create it during writer.open(...)
    parent = os.path.dirname(out_dir) or "."
    os.makedirs(parent, exist_ok=True)

    # Open reader
    reader = rosbag2_py.SequentialReader()
    so_in = rosbag2_py.StorageOptions(uri=args.in_uri, storage_id=args.storage)
    co_in = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(so_in, co_in)

    # Collect topics/types
    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map: Dict[str, str] = {t.name: normalize_ros_type(t.type) for t in topics_and_types}

    # Prepare writer and register topics with same metadata
    writer = rosbag2_py.SequentialWriter()
    so_out = rosbag2_py.StorageOptions(uri=args.out_uri, storage_id=args.storage)
    co_out = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    writer.open(so_out, co_out)

    # Ensure serialization_format present on TopicMetadata when (re)creating
    for t in topics_and_types:
        tm = rosbag2_py.TopicMetadata(
            name=t.name,
            type=normalize_ros_type(t.type),
            serialization_format=t.serialization_format or "cdr",
            offered_qos_profiles=t.offered_qos_profiles if hasattr(t, "offered_qos_profiles") else ""
        )
        writer.create_topic(tm)

    # Cache message classes
    msg_type_cache: Dict[str, object] = {}
    def get_msg_cls(type_str: str):
        t = normalize_ros_type(type_str)
        if t not in msg_type_cache:
            msg_type_cache[t] = get_message(t)
        return msg_type_cache[t]

    # Iterate and rewrite
    ns_to_msg = 1e9  # nanoseconds per second
    gap_ns = int(args.gap_sec * ns_to_msg)
    last_t = None
    last_boundary_t = None
    run_id = 0
    msgs_per_run = 0
    run_counts = []

    print(f"[bag_id_shift] Starting. gap={args.gap_sec:.3f}s, id_offset={args.id_offset}, "
          f"append_ns={args.append_ns}, deleteall_boundary={not args.no_deleteall_boundary}")

    while reader.has_next():
        topic, data, t = reader.read_next()
        type_str = topic_type_map.get(topic)

        # Detect run boundary by *time gap* first (using bag time, not header stamp)
        if last_t is not None:
            if (t - last_t) > gap_ns:
                run_counts.append(msgs_per_run)
                run_id += 1
                msgs_per_run = 0
                last_boundary_t = t
                print(f"[bag_id_shift] Gap boundary -> run {run_id} at {t/1e9:.3f}s")
        last_t = t

        # Deserialize if needed; only deserialize Marker types to speed things up
        if type_str and is_marker_type(type_str):
            msg_cls = get_msg_cls(type_str)
            msg = deserialize_message(data, msg_cls)

            # Detect run boundary by DELETEALL (unless disabled)
            if not args.no_deleteall_boundary and marker_is_delete_all(msg, type_str):
                # Avoid double-increment if we *just* incremented due to a gap
                if last_boundary_t is None or (t - last_boundary_t) > (0.05 * ns_to_msg):
                    run_counts.append(msgs_per_run)
                    run_id += 1
                    msgs_per_run = 0
                    last_boundary_t = t
                    print(f"[bag_id_shift] DELETEALL boundary -> run {run_id} at {t/1e9:.3f}s")

            # Shift IDs / ns for markers
            shift_markers(msg, type_str, run_id, args.id_offset, args.append_ns)
            data = serialize_message(msg)

        # Write (topic name unchanged; timestamps preserved)
        writer.write(topic, data, t)
        msgs_per_run += 1

    run_counts.append(msgs_per_run)
    total_msgs = sum(run_counts)
    print(f"[bag_id_shift] Done. Runs detected: {len(run_counts)} | msgs/run: {run_counts} | total msgs: {total_msgs}")
    print(f"[bag_id_shift] Output bag: {args.out_uri}")

if __name__ == "__main__":
    main()
