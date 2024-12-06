import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

def preprocess_image(image_path):
    """Preprocess the input image for OCR."""
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Resize image to improve OCR accuracy if needed (optional)
    resized = cv2.resize(binary, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return image, resized

def extract_and_display_text(image_path):
    """Extract and display text from an image."""
    # Preprocess the image
    original_image, processed_image = preprocess_image(image_path)

    # Perform OCR with EasyOCR
    results = reader.readtext(processed_image)

    # Visualize results using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    for (bbox, text, prob) in results:
        # Draw bounding box on the image
        pts = np.array([bbox], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(original_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Add the recognized text to the image
        top_left = (int(bbox[0][0]), int(bbox[0][1]))
        plt.text(top_left[0], top_left[1], text, bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='blue')

    plt.axis('off')
    plt.show()

# Path to the image
image_path = './plat1.jpg'  # Replace with your image file

# Extract and display text
extract_and_display_text(image_path)