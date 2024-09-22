# Person Detection with Single Event Trigger for Art Installation

This project uses computer vision and machine learning to detect people and trigger events based on customizable conditions, allowing for flexible, interactive art experiences. The event is triggered only once when conditions are met, with a configurable cooldown period.

## Description

This Python script uses the YOLOv3 object detection model to identify people in real-time using a webcam. It's designed to be integrated into an art installation, with the ability to trigger events based on optional conditions such as the number of people detected and their distance from the camera. The event is triggered only once when conditions are met, preventing rapid re-triggering.

Key features:
- Real-time person detection using a webcam
- Optional distance estimation for detected persons
- Customizable trigger conditions (distance and person count)
- Flexible operation with or without specific trigger conditions
- Single event trigger with configurable cooldown period
- Visual feedback with bounding boxes and optional distance information

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy

## Required Files

The following files must be in the same directory as the script:
- `yolov3.weights`
- `yolov3.cfg`
- `coco.names`

If these files are missing, the script will provide download instructions.

## Installation

1. Clone this repository or download the script.
2. Install the required Python packages:
   ```
   pip install opencv-python numpy
   ```
3. Download the required YOLO files if they're not present:
   - `yolov3.weights`: https://pjreddie.com/media/files/yolov3.weights
   - `yolov3.cfg`: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
   - `coco.names`: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

## Usage

Run the script using Python with optional arguments:

```
python person_detection.py [--focal-length FOCAL_LENGTH] [--trigger-distance DISTANCE] [--trigger-count COUNT] [--cooldown COOLDOWN]
```

All arguments are optional:
- `--focal-length FOCAL_LENGTH`: The calibrated focal length of your camera (required for distance estimation)
- `--trigger-distance DISTANCE`: The distance (in cm) within which people should be detected to trigger the event
- `--trigger-count COUNT`: The number of people required to trigger the event
- `--cooldown COOLDOWN`: The cooldown period (in seconds) between event triggers (default is 5 seconds)

Examples:
1. Basic usage (triggers once for any detected person, with 5-second cooldown):
   ```
   python person_detection.py
   ```

2. Trigger once when 2 or more people are detected, with 10-second cooldown:
   ```
   python person_detection.py --trigger-count 2 --cooldown 10
   ```

3. Trigger once when anyone is within 150 cm of the camera, with 3-second cooldown:
   ```
   python person_detection.py --focal-length 600 --trigger-distance 150 --cooldown 3
   ```

4. Trigger once when 3 or more people are within 200 cm of the camera, with 7-second cooldown:
   ```
   python person_detection.py --focal-length 800 --trigger-distance 200 --trigger-count 3 --cooldown 7
   ```

The webcam feed will open in a new window, showing:
- Bounding boxes around detected persons (green if within trigger distance or if no distance is specified, red otherwise)
- Estimated distance for each detected person (if focal length is provided)
- Current time
- Total number of persons detected
- Number of persons within the trigger distance (if specified)

Press 'q' to quit the application.

## Calibration

For accurate distance estimation, you need to calibrate the focal length of your camera:

1. Place a person at a known distance from the camera (e.g., 100 cm).
2. Measure the width of the person in pixels from the bounding box.
3. Use this formula to calculate the focal length: `focal_length = (pixel_width * known_distance) / known_width`
   where `known_width` is the average width of a person (default is 40 cm in the script).
4. Use this calculated value as the `--focal-length` argument when running the script.

## Customization for Art Installation

To integrate this script into your art installation:
1. Modify the `trigger_event()` function in the script to perform your desired actions when the event is triggered.
2. Adjust the `KNOWN_WIDTH` constant if needed for better distance estimation.
3. Experiment with different combinations of `--focal-length`, `--trigger-distance`, `--trigger-count`, and `--cooldown` to find the best setup for your installation.

## Troubleshooting

- If the webcam doesn't open, ensure it's properly connected and not being used by another application.
- If detection seems off, try adjusting lighting conditions or the confidence threshold in the script.
- If distance estimation is inaccurate, recalibrate the focal length and ensure `KNOWN_WIDTH` is set correctly.
- If the event is triggering too frequently or not frequently enough, adjust the `--cooldown` value.
