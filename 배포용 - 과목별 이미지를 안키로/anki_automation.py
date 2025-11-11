import sys
import subprocess
import pkg_resources

def install_required_packages():
    required_packages = {
        'genanki': 'genanki',
        'pandas': 'pandas',
        'Pillow': 'Pillow',
        'opencv-python': 'opencv-python',
        'numpy': 'numpy'
    }
    
    for package, pip_name in required_packages.items():
        try:
            pkg_resources.require(package)
            print(f"{package} is already installed")
        except pkg_resources.DistributionNotFound:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
            print(f"{package} has been installed")

# Install required packages
print("Checking and installing required packages...")
install_required_packages()

# Import required packages
import genanki
import hashlib
import os
import csv
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

def boxing(image_input):
    """
    Applies bounding boxes to red regions in an image.

    Args:
        image_input: Either a file path (str) or a PIL Image object.

    Returns:
        A PIL Image object with bounding boxes drawn.
    """
    if isinstance(image_input, str):
        # If input is a file path, read the image using cv2
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_input}")
        # Convert OpenCV BGR to RGB for PIL compatibility later
        img_rgb_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_cv2)
    elif isinstance(image_input, Image.Image):
        # If input is a PIL Image, convert it to OpenCV format (BGR)
        img_pil = image_input.convert("RGB") # Ensure RGB format for consistency
        img_rgb_cv2 = np.array(img_pil)
        img = cv2.cvtColor(img_rgb_cv2, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("Input must be a file path (str) or a PIL Image object")

    # Convert to HSV to isolate red regions (use the BGR image for HSV conversion)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color range and create mask
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of red regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes
    boxes = []
    for cnt in contours:
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
        # Ignore tiny dots or noise
        if bbox_w > 30 and bbox_h > 5:
            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

    # Sort boxes by top-to-bottom, then left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # Draw rectangles on the PIL image object
    draw = ImageDraw.Draw(img_pil)
    for x1, y1, x2, y2 in boxes:
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        draw.rectangle([(x1, y1), (x2, y2)], outline="darkred", fill="darkred", width=2)

    return img_pil

def find_blue_boxes(image_input):
    """
    Finds blue regions in an image and returns their bounding boxes.

    Args:
        image_input: Either a file path (str) or a PIL Image object.

    Returns:
        A list of tuples (x1, y1, x2, y2) representing the coordinates of blue boxes,
        sorted from top to bottom.
    """
    if isinstance(image_input, str):
        # If input is a file path, read the image using cv2
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_input}")
    elif isinstance(image_input, Image.Image):
        # If input is a PIL Image, convert it to OpenCV format (BGR)
        img_pil = image_input.convert("RGB")  # Ensure RGB format for consistency
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("Input must be a file path (str) or a PIL Image object")

    # Convert to HSV to isolate blue regions
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define blue color range and create mask
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours of blue regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes
    boxes = []
    for cnt in contours:
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
        # Ignore tiny dots or noise
        if bbox_w > 30 and bbox_h > 5:
            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

    # Sort boxes by top-to-bottom
    boxes = sorted(boxes, key=lambda b: b[1])

    return boxes

def crop_image_to_box(image, box):
    """
    Crops an image to the specified box coordinates.

    Args:
        image: PIL Image object
        box: Tuple of (x1, y1, x2, y2) coordinates

    Returns:
        A PIL Image object containing just the cropped region
    """
    return image.crop(box)

def main():
    # Define the base directory as the current working directory
    base_dir = os.getcwd()
    print(f"Working in directory: {base_dir}")

    # Create a list to store the image data
    image_data_list = []

    # Define the name of the directory to exclude from subjects
    anki_cards_dir_name = "완성된 안키 카드들"

    # Get the list of immediate subdirectories in the base directory (these are the person names)
    try:
        person_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except FileNotFoundError:
        print(f"Base directory not found: {base_dir}")
        return
    
    print("Found person directories:", person_dirs)

    # Process images
    for person_name in person_dirs:
        person_root = os.path.join(base_dir, person_name)

        # Get the list of immediate subdirectories within the person's directory (subjects)
        try:
            subject_dirs = [d for d in os.listdir(person_root) if os.path.isdir(os.path.join(person_root, d))]
        except FileNotFoundError:
            print(f"Person directory not found: {person_root}")
            continue

        for subject_name in subject_dirs:
            # Skip the Anki cards directory
            if subject_name == anki_cards_dir_name:
                continue

            subject_root = os.path.join(person_root, subject_name)
            print(f"Processing {person_name}/{subject_name}")

            # Look for image files directly in the subject directory
            try:
                for file in os.listdir(subject_root):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(subject_root, file)
                        try:
                            original_image = Image.open(image_path)
                            image_data_list.append([person_name, subject_name, file, original_image])
                            print(f"  Added image: {file}")
                        except Exception as e:
                            print(f"Error processing image {image_path}: {e}")
            except FileNotFoundError:
                print(f"Subject directory not found: {subject_root}")
                continue

    if not image_data_list:
        print("No images found to process!")
        return

    print(f"\nFound {len(image_data_list)} images to process")

    # Resize images to target width
    target_width = 1280
    for item in image_data_list:
        original_image = item[3]
        original_width, original_height = original_image.size
        
        if original_width > 0:
            aspect_ratio = original_height / original_width
            new_height = int(target_width * aspect_ratio)
            try:
                resized_image = original_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
                item[3] = resized_image
            except Exception as e:
                print(f"Error resizing image {item[2]}: {e}")

    print("Images resized to target width")

    # Process images for red and blue boxes
    for item in image_data_list:
        original_image = item[3]
        try:
            blue_boxes = find_blue_boxes(original_image)
            
            if blue_boxes:
                print(f"Found {len(blue_boxes)} blue boxes in image: {item[2]}")
                processed_regions = []
                
                for i, blue_box in enumerate(blue_boxes):
                    cropped_image = crop_image_to_box(original_image, blue_box)
                    processed_cropped = boxing(cropped_image)
                    processed_regions.append((i, cropped_image, processed_cropped))
                
                item.append(processed_regions)
            else:
                processed_image = boxing(original_image)
                item.append([(None, original_image, processed_image)])
                
        except Exception as e:
            print(f"Error processing image {item[2]}: {e}")

    # Create output directory for processed images
    output_dir = os.path.join(base_dir, "processed_images_output")
    os.makedirs(output_dir, exist_ok=True)

    # Save processed images and create CSV
    csv_output_path = os.path.join(output_dir, "image_pairs.csv")
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for item in image_data_list:
            original_filename = item[2]
            processed_regions = item[4]
            
            for idx, orig_crop, proc_crop in processed_regions:
                name, ext = os.path.splitext(original_filename)
                region_suffix = f"_region{idx+1}" if idx is not None else ""
                
                processed_img_save_name = f"{name}{region_suffix}-1_processed{ext}"
                original_img_save_name = f"{name}{region_suffix}-1_original{ext}"
                
                processed_img_save_path = os.path.join(output_dir, processed_img_save_name)
                original_img_save_path = os.path.join(output_dir, original_img_save_name)
                
                proc_crop.save(processed_img_save_path)
                orig_crop.save(original_img_save_path)
                csv_writer.writerow([processed_img_save_name, original_img_save_name])

    # Group data by person and subject
    grouped_data = {}
    for item in image_data_list:
        person_name = item[0]
        subject_name = item[1]
        
        if person_name not in grouped_data:
            grouped_data[person_name] = {}
        if subject_name not in grouped_data[person_name]:
            grouped_data[person_name][subject_name] = []
            
        grouped_data[person_name][subject_name].append(item)

    # Create Anki decks
    base_model_id = 1234567890
    decks_to_create = []

    for person_name, subjects_data in grouped_data.items():
        for subject_name, image_items in subjects_data.items():
            subject_hash = int(hashlib.sha256(subject_name.encode('utf-8')).hexdigest(), 16) % (10**9)
            model_id = base_model_id + subject_hash

            my_model = genanki.Model(
                model_id,
                'Simple Image Card',
                fields=[
                    {'name': 'Processed Image'},
                    {'name': 'Original Image'},
                    {'name': 'Original Filename'},
                ],
                templates=[{
                    'name': 'Card 1',
                    'qfmt': '{{Processed Image}}',
                    'afmt': '{{FrontSide}}<hr id="answer">{{Original Image}}<br>Original Filename: {{Original Filename}}',
                }])

            my_deck = genanki.Deck(
                abs(hash(subject_name)),
                '자동으로 만든 덱::자동으로 만든 ' + subject_name)

            decks_to_create.append((my_deck, my_model, image_items, person_name, subject_name))

    # Create notes and package decks
    anki_output_dir = os.path.join(base_dir, "완성된 안키 카드들")
    os.makedirs(anki_output_dir, exist_ok=True)

    for my_deck, my_model, image_items, person_name, subject_name in decks_to_create:
        # Add notes to deck
        for item in image_items:
            original_filename = item[2]
            processed_regions = item[4]
            
            if processed_regions[0][0] is not None:
                processed_html = []
                original_html = []
                
                for idx, orig_crop, proc_crop in processed_regions:
                    name, ext = os.path.splitext(original_filename)
                    region_suffix = f"_region{idx+1}"
                    
                    processed_img_filename = f"{name}{region_suffix}-1_processed{ext}"
                    original_img_filename = f"{name}{region_suffix}-1_original{ext}"
                    
                    processed_html.append(f'<img src="{processed_img_filename}">')
                    original_html.append(f'<img src="{original_img_filename}">')
                
                my_note = genanki.Note(
                    model=my_model,
                    fields=[
                        '<br>'.join(processed_html),
                        '<br>'.join(original_html),
                        original_filename
                    ])
                
            else:
                processed_img_filename = f"{original_filename.rsplit('.', 1)[0]}-1_processed.{original_filename.rsplit('.', 1)[1]}"
                original_img_filename = f"{original_filename.rsplit('.', 1)[0]}-1_original.{original_filename.rsplit('.', 1)[1]}"
                
                my_note = genanki.Note(
                    model=my_model,
                    fields=[
                        f'<img src="{processed_img_filename}">',
                        f'<img src="{original_img_filename}">',
                        original_filename
                    ])
            
            my_deck.add_note(my_note)

        # Package deck with media files
        base_apkg_filename = f"완성된 {subject_name}.apkg"
        apkg_output_path = os.path.join(anki_output_dir, base_apkg_filename)

        counter = 1
        while os.path.exists(apkg_output_path):
            name, ext = os.path.splitext(base_apkg_filename)
            apkg_filename = f"{name}-{counter}{ext}"
            apkg_output_path = os.path.join(anki_output_dir, apkg_filename)
            counter += 1

        print(f"Packaging deck: {subject_name} for {person_name}")

        media_files = []
        for item in image_items:
            original_filename = item[2]
            processed_regions = item[4]

            for idx, orig_crop, proc_crop in processed_regions:
                name, ext = os.path.splitext(original_filename)
                region_suffix = f"_region{idx+1}" if idx is not None else ""
                
                processed_img_filename = f"{name}{region_suffix}-1_processed{ext}"
                original_img_filename = f"{name}{region_suffix}-1_original{ext}"

                processed_img_path = os.path.join(output_dir, processed_img_filename)
                original_img_path = os.path.join(output_dir, original_img_filename)

                media_files.append(processed_img_path)
                media_files.append(original_img_path)

        try:
            genanki.Package(my_deck, media_files=media_files).write_to_file(apkg_output_path)
            print(f"Successfully created Anki package: {apkg_output_path}")
        except Exception as e:
            print(f"Error creating Anki package: {e}")

    # Clean up processed images
    print("\nCleaning up processed images...")
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print("Cleaned up processed images directory")
        except Exception as e:
            print(f"Error cleaning up processed images: {e}")

if __name__ == "__main__":
    main()