#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
cd /home/ubuntu/laser_detection
python3 inference_stream_2.py
