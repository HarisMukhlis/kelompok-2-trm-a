import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import torch

# Check CUDA availability
is_cuda_available = torch.cuda.is_available()
frame_skip = 5 if is_cuda_available else 30  # Use 5 if CUDA is available, otherwise 30

# Display GPU or CPU info
device = "GPU" if is_cuda_available else "CPU"
print(f"Running on: {device}")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=is_cuda_available)

def extract_text_and_bounding_boxes(frame):
    """Extract text and bounding boxes from a video frame."""
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edge = cv2.Canny(gray, threshold1=100, threshold2=200)

    # _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Use EasyOCR to detect and extract text
    results = reader.readtext(gray)

    # Return results (text and bounding boxes)
    return results

def display_frame_with_text(frame, results):
    """Display frame with text and bounding boxes using matplotlib."""
    # Convert frame from BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Plot the frame
    plt.imshow(frame_rgb)
    plt.axis('off')

    # Draw bounding boxes and text
    for bbox, text, _ in results:
        # Extract bounding box coordinates
        (x_min, y_min), _, (x_max, y_max), _ = bbox

        # Draw the bounding box
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min],
                 color='green', linewidth=2)

        # Display the text
        plt.text(x_min, y_min - 10, text, color='green', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # Show the plot
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()

def main():
    # Initialize camera
    frame_skip = 30  # Process one out of every 5 frames
    frame_count = 0

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            results = extract_text_and_bounding_boxes(frame)

        display_frame_with_text(frame, results)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# Run the main function
main()