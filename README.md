# Person Detection for Art Installation

This project uses computer vision and machine learning to detect when a person approaches the installation, allowing for dynamic, interactive art experiences.

## Description

This Python script uses the YOLOv3 object detection model to identify people in real-time using a webcam. It's designed to be integrated into an art installation, triggering events when a person is detected.

Key features:
- Real-time person detection using a webcam
- Display of current time and number of persons detected
- Visual bounding boxes around detected persons

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
   pip install -r requirements.txt
   ```
3. Download the required YOLO files if they're not present:
   - `yolov3.weights`: https://pjreddie.com/media/files/yolov3.weights
   - `yolov3.cfg`: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
   - `coco.names`: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

## Usage

Run the script using Python:

```
python main.py
```

The webcam feed will open in a new window, showing:
- Bounding boxes around detected persons
- Current time
- Number of persons detected

Press 'q' to quit the application.

## Customization for Art Installation

To integrate this script into your art installation:
1. Modify the confidence threshold (currently set to 0.5) to adjust sensitivity.
2. Add code to trigger your installation's events when a person is detected.
3. Customize the visual output or remove it if not needed for the installation.

## Troubleshooting

- If the webcam doesn't open, ensure it's properly connected and not being used by another application.
- If detection seems off, try adjusting lighting conditions or the confidence threshold.
