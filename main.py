import cv2
import numpy as np
from datetime import datetime
import os
import argparse
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Person detection with optional trigger conditions for art installations.")
parser.add_argument("--focal-length", type=float, help="Focal length of the camera (calibrated)")
parser.add_argument("--trigger-distance", type=float, help="Distance to trigger the event (in cm)")
parser.add_argument("--trigger-count", type=int, help="Number of people required to trigger the event")
parser.add_argument("--cooldown", type=float, default=5.0, help="Cooldown period between triggers (in seconds)")
args = parser.parse_args()

FOCAL_LENGTH = args.focal_length
TRIGGER_DISTANCE = args.trigger_distance
TRIGGER_COUNT = args.trigger_count
COOLDOWN_PERIOD = args.cooldown

print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir()}")

# Check if files exist
required_files = ["yolov3.weights", "yolov3.cfg", "coco.names"]
for file in required_files:
    if not os.path.exists(file):
        print(f"Error: {file} not found in the current directory.")
        if file == "coco.names":
            print("You can download coco.names using this command:")
            print("wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
        elif file == "yolov3.weights":
            print("You can download yolov3.weights using this command:")
            print("wget https://pjreddie.com/media/files/yolov3.weights")
        elif file == "yolov3.cfg":
            print("You can download yolov3.cfg using this command:")
            print("wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
    else:
        print(f"{file} found.")

# Constants for distance estimation
KNOWN_WIDTH = 40  # cm (approximate width of a person)

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth if focalLength else None

# Load YOLOv3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    raise Exception("Error: Could not open webcam.")
else:
    print("Webcam opened successfully.")

def trigger_event():
    print("Event triggered!")
    # Add your event triggering code here

last_trigger_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to display
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count persons, draw bounding boxes, and estimate distances
    person_count = 0
    close_person_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            
            # Estimate distance if focal length is provided
            distance = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, w) if FOCAL_LENGTH else None
            
            # Determine if person is within trigger distance
            is_close = distance <= TRIGGER_DISTANCE if TRIGGER_DISTANCE and distance else True
            
            # Draw bounding box
            color = (0, 255, 0) if is_close else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Display distance if available
            if distance:
                label = f"{distance:.2f}cm"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            person_count += 1
            if is_close:
                close_person_count += 1

    # Check if event should be triggered
    current_time = time.time()
    should_trigger = True
    if TRIGGER_COUNT is not None:
        should_trigger = should_trigger and (close_person_count >= TRIGGER_COUNT)
    
    if should_trigger and (current_time - last_trigger_time > COOLDOWN_PERIOD):
        trigger_event()
        last_trigger_time = current_time

    # Get current time for display
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Display information on the frame
    cv2.putText(frame, f"Time: {current_time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Persons detected: {person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if TRIGGER_DISTANCE:
        cv2.putText(frame, f"Persons within trigger distance: {close_person_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the result
    cv2.imshow("Webcam Person Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()