import os
import glob
import cv2
import csv
import time
from ultralytics import YOLO
import natsort 


# --- Configuration ---
SAURON_MODEL_PATH = r"/home/sauron/Downloads/SAURON.pt" 
LWIR_IMAGE_DIR = r"/media/sauron/SAURON/Out"  
RGB_IMAGE_DIR = r"/media/sauron/SAURON/forDistro/1Downlooking/RGB"    
OUTPUT_CSV_FILE = "sauron_detections_log.csv"         # File to save detection results

# Image file extensions to look for (adjust if needed)
IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp')

# Detection & Transformation Settings
CONFIDENCE_THRESHOLD = 0.5  
# **IMPORTANT**: Verify these flip flags based on your camera setup (see Enhancement 6)
NEEDS_HORIZONTAL_FLIP = True
NEEDS_VERTICAL_FLIP = True

# CSV File Headers
CSV_FIELDNAMES = [
    "timestamp", "lwir_source", "rgb_source", "class_id", "class_name",
    "confidence", "lwir_bbox_xyxy", "rgb_bbox_xyxy"
]

# --- Helper Functions ---

def load_yolo_model(model_path):
    """Loads the YOLO model with error handling."""
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded YOLO model from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def load_image(image_path):
    """Loads an image using OpenCV with error handling."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image (cv2.imread returned None): {image_path}")
            return None
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def transform_coordinates(lwir_box, lwir_shape, rgb_shape, flip_horizontal, flip_vertical):
    """Transforms bounding box coordinates from LWIR to RGB space dynamically."""
    try:
        lwir_h, lwir_w = lwir_shape[:2]
        rgb_h, rgb_w = rgb_shape[:2]

        if lwir_w == 0 or lwir_h == 0 or rgb_w == 0 or rgb_h == 0:
            print("Error: Invalid image dimensions for transformation.")
            return None

        h_scale = rgb_w / lwir_w
        v_scale = rgb_h / lwir_h

        orig_x1, orig_y1, orig_x2, orig_y2 = map(int, lwir_box)

        x1_s  = int(orig_x1 * h_scale) 
        y1_s = int(orig_y1 * v_scale)
        x2_s = int(orig_x2 * h_scale)
        y2_s = int(orig_y2 * v_scale)

        if flip_horizontal:
            final_x1 = rgb_w - x2_s
            final_x2 = rgb_w - x1_s
        else:
            final_x1 = x1_s
            final_x2 = x2_s

        if flip_vertical:
            final_y1 = rgb_h - y2_s
            final_y2 = rgb_h - y1_s
        else:
            final_y1 = y1_s
            final_y2 = y2_s

        # Clip coordinates to be within RGB image bounds
        final_x1 = max(0, min(final_x1, rgb_w - 1))
        final_y1 = max(0, min(final_y1, rgb_h - 1))
        final_x2 = max(0, min(final_x2, rgb_w - 1))
        final_y2 = max(0, min(final_y2, rgb_h - 1))

        # Ensure x1 < x2 and y1 < y2 after potential flipping/clipping
        if final_x1 >= final_x2 or final_y1 >= final_y2:
             print(f"Warning: Degenerate bounding box after transformation: {[final_x1, final_y1, final_x2, final_y2]}")
             return None

        return [final_x1, final_y1, final_x2, final_y2]

    except Exception as e:
        print(f"Error during coordinate transformation: {e}")
        return None

def initialize_csv_writer(output_file, fieldnames):
    """Initializes a CSV writer, writing header if file is new."""
    try:
        file_exists = os.path.isfile(output_file)
        csv_file = open(output_file, mode='a', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(output_file) == 0:
            writer.writeheader()
            print(f"Initialized CSV log file: {output_file}")
        return writer, csv_file
    except IOError as e:
        print(f"Error opening or initializing CSV file {output_file}: {e}")
        return None, None

def save_detection_csv(writer, detection_data):
    """Saves a single detection record to the CSV file."""
    if writer:
        try:
            writer.writerow(detection_data)
        except IOError as e:
            print(f"Error writing detection to CSV: {e}")
        except Exception as e:
            print(f"Unexpected error writing detection data: {e} - Data: {detection_data}")

# --- Main Processing Logic ---

if __name__ == "__main__":
    # 1. Load Model
    sauron_model = load_yolo_model(SAURON_MODEL_PATH)
    if sauron_model is None:
        exit()

    # 2. Initialize CSV Writer
    csv_writer, csv_file_handle = initialize_csv_writer(OUTPUT_CSV_FILE, CSV_FIELDNAMES)

    # 3. Find and Sort LWIR and RGB Images
    lwir_image_paths = []
    for ext in IMAGE_EXTENSIONS:
        lwir_image_paths.extend(glob.glob(os.path.join(LWIR_IMAGE_DIR, ext)))

    rgb_image_paths = []
    for ext in IMAGE_EXTENSIONS:
        rgb_image_paths.extend(glob.glob(os.path.join(RGB_IMAGE_DIR, ext)))

    try:
        lwir_image_paths = natsort.natsorted(lwir_image_paths)
        rgb_image_paths = natsort.natsorted(rgb_image_paths)
    except NameError:
        print("Warning: natsort library not found (pip install natsort). Using standard sort.")
        lwir_image_paths.sort()
        rgb_image_paths.sort()


    # 4. Validate Image Counts
    if not lwir_image_paths:
        print(f"Error: No LWIR images found in {LWIR_IMAGE_DIR}")
        if csv_file_handle: csv_file_handle.close()
        exit()
    if not rgb_image_paths:
        print(f"Error: No RGB images found in {RGB_IMAGE_DIR}")
        if csv_file_handle: csv_file_handle.close()
        exit()

    if len(lwir_image_paths)!= len(rgb_image_paths):
        print(f"Error: Mismatched number of images! Found {len(lwir_image_paths)} LWIR images and {len(rgb_image_paths)} RGB images.")
        print("Cannot perform sequential pairing. Please ensure directories contain corresponding images.")
        if csv_file_handle: csv_file_handle.close()
        exit()

    print(f"Found {len(lwir_image_paths)} corresponding LWIR and RGB images.")

    # 5. Process Each Image Pair Sequentially
    for index, lwir_path in enumerate(lwir_image_paths):
        rgb_path = rgb_image_paths[index] # Get the corresponding RGB path by index

        print(f"\nProcessing Pair {index + 1}/{len(lwir_image_paths)}:")
        print(f"  LWIR: {os.path.basename(lwir_path)}")
        print(f"  RGB:  {os.path.basename(rgb_path)}")

        # Load LWIR image
        lwir_image = load_image(lwir_path)
        if lwir_image is None:
            continue # Skip to next pair

        # Load RGB image
        rgb_image = load_image(rgb_path)
        if rgb_image is None:
            print(f"  Skipping: Failed to load RGB image {os.path.basename(rgb_path)}")
            continue

        # Perform detection on LWIR image
        try:
            results = sauron_model(lwir_image, verbose=False)
        except Exception as e:
            print(f"  Error during YOLO inference: {e}")
            continue

        # Process detections for this image
        detections_drawn = 0
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_indices = result.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes):
                confidence = confidences[i]

                if confidence >= CONFIDENCE_THRESHOLD:
                    transformed_box = transform_coordinates(
                        box, lwir_image.shape[:2], rgb_image.shape[:2],
                        NEEDS_HORIZONTAL_FLIP, NEEDS_VERTICAL_FLIP
                    )

                    if transformed_box:
                    
                    	# --- START: Add Padding to Bounding Box ---
                        padding_pixels = 100 # Adjust this value as needed (e.g., 5, 10, 15)
                        rgb_h, rgb_w = rgb_image.shape[:2] # Get RGB dimensions for clipping

                        # Original transformed coordinates
                        tx1, ty1, tx2, ty2 = transformed_box

                        # Apply padding
                        padded_x1 = tx1 - padding_pixels
                        padded_y1 = ty1 - padding_pixels
                        padded_x2 = tx2 + padding_pixels
                        padded_y2 = ty2 + padding_pixels

                        # Clip padded coordinates to stay within image boundaries
                        padded_x1 = max(0, padded_x1)
                        padded_y1 = max(0, padded_y1)
                        padded_x2 = min(rgb_w - 1, padded_x2)
                        padded_y2 = min(rgb_h - 1, padded_y2)

                        # Ensure box is still valid after padding/clipping
                        if padded_x1 >= padded_x2 or padded_y1 >= padded_y2:
                            print("  Warning: Bounding box became invalid after padding. Using original.")
                            # Fallback to original transformed box if padding makes it invalid
                            draw_x1, draw_y1, draw_x2, draw_y2 = tx1, ty1, tx2, ty2
                        else:
                            draw_x1, draw_y1, draw_x2, draw_y2 = (padded_x1), (padded_y1), (padded_x2) , (padded_y2)
                            
                        detections_drawn += 1
                        # final_x1, final_y1, final_x2, final_y2 = transformed_box # Original line (replaced)
                        class_id = int(class_indices[i])
                        class_name = sauron_model.names[class_id]
                        label = f"{class_name}: {confidence:.2f}"

                        # Draw bounding box using the PADDED coordinates
                        cv2.rectangle(rgb_image, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
                        # Put text label relative to the PADDED box's top-left corner
                        cv2.putText(rgb_image, label, (draw_x1, draw_y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                       # Prepare data for CSV logging (still log the ORIGINAL transformed box)
                        detection_record = {
                            "timestamp": time.time(),
                            "lwir_source": lwir_path,
                            "rgb_source": rgb_path,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": f"{confidence:.4f}",
                            "lwir_bbox_xyxy": [int(coord) for coord in box],
                            "rgb_bbox_xyxy": transformed_box
                        }
                        save_detection_csv(csv_writer, detection_record)
                    else:
                        print(f"  Warning: Coordinate transformation failed for a detection with confidence {confidence:.2f}")

        if detections_drawn > 0:
            print(f"  Displayed {detections_drawn} detections passing threshold.")
        else:
            print("  No detections passed the threshold for this image pair.")

        try:
            # Define desired display width (adjust as needed)
            display_width = 1280
            h, w = rgb_image.shape[:2]

            # Calculate aspect ratio and new height
            aspect_ratio = h / w
            display_height = int(display_width * aspect_ratio)

            # Resize the image for display
            display_image = cv2.resize(rgb_image, (display_width, display_height), interpolation=cv2.INTER_AREA)

        except Exception as e:
            print(f"  Warning: Could not resize image for display. Showing original size. Error: {e}")
            display_image = rgb_image # Fallback to original if resize fails

        # Display the RESIZED image
        cv2.imshow("SAURON Detection Result (Press ESC to quit, any other key for next)", display_image) # Use display_image here
        key = cv2.waitKey(1)

        if key == 27:
            print("ESC key pressed. Exiting.")
            break

	
    # 6. Cleanup
    cv2.destroyAllWindows()
    if csv_file_handle:
        csv_file_handle.close()
        print(f"Closed CSV log file: {OUTPUT_CSV_FILE}")

    print("\nProcessing complete.")
