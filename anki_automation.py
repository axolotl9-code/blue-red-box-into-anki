#!/usr/bin/env python3#!/usr/bin/env python3

"""import genanki

이미지 처리 및 안키 카드 생성 자동화 스크립트import hashlib

사용법: python anki_automation.py [입력_디렉토리]import os

"""import csv

import shutil

import genankifrom PIL import Image, ImageDraw

import hashlibimport cv2

import osimport numpy as np

import csv

import shutildef boxing(image_input):

from PIL import Image, ImageDraw    """

import cv2    Applies bounding boxes to red regions in an image.

import numpy as np

import argparse    Args:

import sys        image_input: Either a file path (str) or a PIL Image object.



def boxing(image_input):    Returns:

    """        A PIL Image object with bounding boxes drawn.

    이미지에서 빨간색 영역을 찾아 박스 처리합니다.    """

    if isinstance(image_input, str):

    Args:        # If input is a file path, read the image using cv2

        image_input: 파일 경로(str) 또는 PIL Image 객체        img = cv2.imread(image_input)

        if img is None:

    Returns:            raise FileNotFoundError(f"Image not found at {image_input}")

        박스가 그려진 PIL Image 객체        # Convert OpenCV BGR to RGB for PIL compatibility later

    """        img_rgb_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if isinstance(image_input, str):        img_pil = Image.fromarray(img_rgb_cv2)

        img = cv2.imread(image_input)    elif isinstance(image_input, Image.Image):

        if img is None:        # If input is a PIL Image, convert it to OpenCV format (BGR)

            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_input}")        img_pil = image_input.convert("RGB") # Ensure RGB format for consistency

        img_rgb_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        img_rgb_cv2 = np.array(img_pil)

        img_pil = Image.fromarray(img_rgb_cv2)        img = cv2.cvtColor(img_rgb_cv2, cv2.COLOR_RGB2BGR)

    elif isinstance(image_input, Image.Image):    else:

        img_pil = image_input.convert("RGB")        raise TypeError("Input must be a file path (str) or a PIL Image object")

        img_rgb_cv2 = np.array(img_pil)

        img = cv2.cvtColor(img_rgb_cv2, cv2.COLOR_RGB2BGR)    # Convert to HSV to isolate red regions (use the BGR image for HSV conversion)

    else:    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        raise TypeError("입력은 파일 경로(str) 또는 PIL Image 객체여야 합니다")

    # Define red color range and create mask

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    lower_red1 = np.array([0, 70, 50])

    upper_red1 = np.array([10, 255, 255])

    # 빨간색 범위 정의 및 마스크 생성    lower_red2 = np.array([170, 70, 50])

    lower_red1 = np.array([0, 70, 50])    upper_red2 = np.array([180, 255, 255])

    upper_red1 = np.array([10, 255, 255])    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 70, 50])    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    upper_red2 = np.array([180, 255, 255])    mask = cv2.bitwise_or(mask1, mask2)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)    # Find contours of red regions

    mask = cv2.bitwise_or(mask1, mask2)    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # Extract bounding boxes

    boxes = []

    boxes = []    for cnt in contours:

    for cnt in contours:        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)        # Ignore tiny dots or noise

        if bbox_w > 30 and bbox_h > 5:  # 작은 노이즈 제거        if bbox_w > 30 and bbox_h > 5:

            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))



    # 위에서 아래로, 왼쪽에서 오른쪽으로 정렬    # Sort boxes by top-to-bottom, then left-to-right

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))



    draw = ImageDraw.Draw(img_pil)    # Draw rectangles on the PIL image object

    for x1, y1, x2, y2 in boxes:    draw = ImageDraw.Draw(img_pil)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)    for x1, y1, x2, y2 in boxes:

        draw.rectangle([(x1, y1), (x2, y2)], outline="darkred", fill="darkred", width=2)        # Convert coordinates to integers

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return img_pil        draw.rectangle([(x1, y1), (x2, y2)], outline="darkred", fill="darkred", width=2)



def find_blue_boxes(image_input):    return img_pil

    """

    이미지에서 파란색 영역을 찾아 좌표를 반환합니다.def find_blue_boxes(image_input):

    """

    Args:    Finds blue regions in an image and returns their bounding boxes.

        image_input: 파일 경로(str) 또는 PIL Image 객체

    Args:

    Returns:        image_input: Either a file path (str) or a PIL Image object.

        파란색 박스의 좌표 목록 [(x1, y1, x2, y2), ...], 위에서 아래로 정렬됨

    """    Returns:

    if isinstance(image_input, str):        A list of tuples (x1, y1, x2, y2) representing the coordinates of blue boxes,

        img = cv2.imread(image_input)        sorted from top to bottom.

        if img is None:    """

            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_input}")    if isinstance(image_input, str):

    elif isinstance(image_input, Image.Image):        # If input is a file path, read the image using cv2

        img_pil = image_input.convert("RGB")        img = cv2.imread(image_input)

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)        if img is None:

    else:            raise FileNotFoundError(f"Image not found at {image_input}")

        raise TypeError("입력은 파일 경로(str) 또는 PIL Image 객체여야 합니다")    elif isinstance(image_input, Image.Image):

        # If input is a PIL Image, convert it to OpenCV format (BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        img_pil = image_input.convert("RGB")  # Ensure RGB format for consistency

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 파란색 범위 정의 및 마스크 생성    else:

    lower_blue = np.array([100, 50, 50])        raise TypeError("Input must be a file path (str) or a PIL Image object")

    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)    # Convert to HSV to isolate blue regions

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define blue color range and create mask

    boxes = []    # Blue typically has a hue value around 120 in HSV

    for cnt in contours:    lower_blue = np.array([100, 50, 50])

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)    upper_blue = np.array([140, 255, 255])

        if bbox_w > 30 and bbox_h > 5:  # 작은 노이즈 제거    mask = cv2.inRange(hsv, lower_blue, upper_blue)

            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

    # Find contours of blue regions

    # 위에서 아래로 정렬    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = sorted(boxes, key=lambda b: b[1])

    # Extract bounding boxes

    return boxes    boxes = []

    for cnt in contours:

def crop_image_to_box(image, box):        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)

    """        # Ignore tiny dots or noise (adjust these thresholds as needed)

    이미지를 지정된 박스 영역으로 자릅니다.        if bbox_w > 30 and bbox_h > 5:

            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

    Args:

        image: PIL Image 객체    # Sort boxes by top-to-bottom

        box: 좌표 튜플 (x1, y1, x2, y2)    boxes = sorted(boxes, key=lambda b: b[1])



    Returns:    return boxes

        잘린 영역의 PIL Image 객체

    """def crop_image_to_box(image, box):

    return image.crop(box)    """

    Crops an image to the specified box coordinates.

def process_images(base_dir=".", target_width=1280):

    """    Args:

    메인 처리 함수: 이미지를 처리하고 안키 카드를 생성합니다.        image: PIL Image object

            box: Tuple of (x1, y1, x2, y2) coordinates

    Args:

        base_dir: 입력 디렉토리 경로 (기본값: 현재 디렉토리)    Returns:

        target_width: 이미지 리사이징할 목표 너비 (기본값: 1280)        A PIL Image object containing just the cropped region

    """    """

    print(f"처리 시작: {base_dir}")    return image.crop(box)

    

    image_data_list = []def process_images(base_dir=".", target_width=1280):

    anki_cards_dir_name = "완성된 안키 카드들"    """Main function to process images and create Anki cards"""

    # Create a list to store the specified information for each image

    try:    image_data_list = []

        person_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]    anki_cards_dir_name = "완성된 안키 카드들"

    except FileNotFoundError:

        print(f"디렉토리를 찾을 수 없습니다: {base_dir}")    # Get the list of immediate subdirectories (these are person names)

        return    try:

        person_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    print("이미지 수집 및 전처리 중...")    except FileNotFoundError:

    for person_name in person_dirs:        print(f"Base directory not found: {base_dir}")

        person_root = os.path.join(base_dir, person_name)        return

        

        try:    # Process each person's directory

            subject_dirs = [d for d in os.listdir(person_root) if os.path.isdir(os.path.join(person_root, d))]    for person_name in person_dirs:

        except FileNotFoundError:        person_root = os.path.join(base_dir, person_name)

            print(f"디렉토리를 찾을 수 없습니다: {person_root}")        

            continue        # Get subject directories

        try:

        for subject_name in subject_dirs:            subject_dirs = [d for d in os.listdir(person_root) if os.path.isdir(os.path.join(person_root, d))]

            if subject_name == anki_cards_dir_name:        except FileNotFoundError:

                continue            print(f"Person directory not found: {person_root}")

            continue

            subject_root = os.path.join(person_root, subject_name)

        # Process each subject directory

            try:        for subject_name in subject_dirs:

                for file in os.listdir(subject_root):            if subject_name == anki_cards_dir_name:

                    image_path = os.path.join(subject_root, file)                continue

                    if os.path.isfile(image_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):

                        try:            subject_root = os.path.join(person_root, subject_name)

                            original_image = Image.open(image_path)

                            original_width, original_height = original_image.size            # Process images in the subject directory

                            if original_width > 0:            try:

                                aspect_ratio = original_height / original_width                for file in os.listdir(subject_root):

                                new_height = int(target_width * aspect_ratio)                    image_path = os.path.join(subject_root, file)

                                resized_image = original_image.resize((target_width, new_height), Image.Resampling.LANCZOS)                    if os.path.isfile(image_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):

                            else:                        try:

                                resized_image = original_image                            # Open and resize image

                                print(f"경고: 이미지의 너비가 0입니다: {file}")                            original_image = Image.open(image_path)

                            original_width, original_height = original_image.size

                            image_data_list.append([                            if original_width > 0:

                                person_name,                                aspect_ratio = original_height / original_width

                                subject_name,                                new_height = int(target_width * aspect_ratio)

                                os.path.basename(image_path),                                resized_image = original_image.resize((target_width, new_height), Image.Resampling.LANCZOS)

                                resized_image                            else:

                            ])                                resized_image = original_image

                            print(f"이미지 처리 중: {file}")                                print(f"Warning: Original image width is zero for {file}")



                        except Exception as e:                            # Store image data

                            print(f"이미지 처리 오류 {image_path}: {e}")                            image_data_list.append([

            except FileNotFoundError:                                person_name,

                print(f"디렉토리를 찾을 수 없습니다: {subject_root}")                                subject_name,

                continue                                os.path.basename(image_path),

                                resized_image

    if not image_data_list:                            ])

        print("처리할 이미지가 없습니다.")

        return                        except Exception as e:

                            print(f"Error processing image {image_path}: {e}")

    print("\n영역 검출 중...")            except FileNotFoundError:

    output_dir = os.path.join(base_dir, "processed_images_output")                print(f"Subject directory not found: {subject_root}")

    os.makedirs(output_dir, exist_ok=True)                continue



    grouped_data = {}    # Create output directory for processed images

        output_dir = os.path.join(base_dir, "processed_images_output")

    for item in image_data_list:    os.makedirs(output_dir, exist_ok=True)

        original_image = item[3]

        try:    # Process images and create Anki cards

            blue_boxes = find_blue_boxes(original_image)    grouped_data = {}

                

            if blue_boxes:    # Process each image and detect regions

                print(f"{len(blue_boxes)}개의 파란색 영역 발견: {item[2]}")    for item in image_data_list:

                processed_regions = []        original_image = item[3]

                for i, blue_box in enumerate(blue_boxes, 1):        try:

                    cropped_image = crop_image_to_box(original_image, blue_box)            blue_boxes = find_blue_boxes(original_image)

                    processed_cropped = boxing(cropped_image)            

                    processed_regions.append((i-1, cropped_image, processed_cropped))            if blue_boxes:

                item.append(processed_regions)                processed_regions = []

            else:                for i, blue_box in enumerate(blue_boxes):

                print(f"전체 이미지 처리 중: {item[2]}")                    cropped_image = crop_image_to_box(original_image, blue_box)

                processed_image = boxing(original_image)                    processed_cropped = boxing(cropped_image)

                item.append([(None, original_image, processed_image)])                    processed_regions.append((i, cropped_image, processed_cropped))

                        item.append(processed_regions)

        except Exception as e:            else:

            print(f"이미지 처리 오류 {item[2]}: {e}")                processed_image = boxing(original_image)

                item.append([(None, original_image, processed_image)])

    print("\n데이터 그룹화 중...")        

    for item in image_data_list:        except Exception as e:

        person_name = item[0]            print(f"Error processing image {item[2]}: {e}")

        subject_name = item[1]

    # Group data by person and subject

        if person_name not in grouped_data:    for item in image_data_list:

            grouped_data[person_name] = {}        person_name = item[0]

        if subject_name not in grouped_data[person_name]:        subject_name = item[1]

            grouped_data[person_name][subject_name] = []

                if person_name not in grouped_data:

        grouped_data[person_name][subject_name].append(item)            grouped_data[person_name] = {}

        if subject_name not in grouped_data[person_name]:

    print("\n안키 덱 생성 중...")            grouped_data[person_name][subject_name] = []

    decks_to_create = []        

    base_model_id = 1234567890        grouped_data[person_name][subject_name].append(item)



    for person_name, subjects_data in grouped_data.items():    # Create Anki decks

        for subject_name, image_items in subjects_data.items():    decks_to_create = []

            print(f"덱 준비 중: {subject_name} ({person_name})")    base_model_id = 1234567890

            subject_hash = int(hashlib.sha256(subject_name.encode('utf-8')).hexdigest(), 16) % (10**9)

            model_id = base_model_id + subject_hash    for person_name, subjects_data in grouped_data.items():

        for subject_name, image_items in subjects_data.items():

            my_model = genanki.Model(            subject_hash = int(hashlib.sha256(subject_name.encode('utf-8')).hexdigest(), 16) % (10**9)

                model_id,            model_id = base_model_id + subject_hash

                'Simple Image Card',

                fields=[            my_model = genanki.Model(

                    {'name': 'Original Image'},                model_id,

                    {'name': 'Processed Image'},                'Simple Image Card',

                    {'name': 'Original Filename'},                fields=[

                ],                    {'name': 'Original Image'},

                templates=[{                    {'name': 'Processed Image'},

                    'name': 'Card 1',                    {'name': 'Original Filename'},

                    'qfmt': '{{Original Image}}',                ],

                    'afmt': '{{FrontSide}}<hr id="answer">{{Processed Image}}<br>Original Filename: {{Original Filename}}',                templates=[{

                }])                    'name': 'Card 1',

                    'qfmt': '{{Original Image}}',

            my_deck = genanki.Deck(                    'afmt': '{{FrontSide}}<hr id="answer">{{Processed Image}}<br>Original Filename: {{Original Filename}}',

                abs(hash(subject_name)),                }])

                '자동으로 만든 덱::자동으로 만든 '+subject_name)

            my_deck = genanki.Deck(

            decks_to_create.append((my_deck, my_model, image_items, person_name, subject_name))                abs(hash(subject_name)),

                '자동으로 만든 덱::자동으로 만든 '+subject_name)

    print("\n안키 패키지 생성 중...")

    anki_output_dir = os.path.join(base_dir, anki_cards_dir_name)            decks_to_create.append((my_deck, my_model, image_items, person_name, subject_name))

    os.makedirs(anki_output_dir, exist_ok=True)

    # Save processed images and create Anki packages

    for my_deck, my_model, image_items, person_name, subject_name in decks_to_create:    anki_output_dir = os.path.join(base_dir, anki_cards_dir_name)

        for item in image_items:    os.makedirs(anki_output_dir, exist_ok=True)

            original_filename = item[2]

            processed_regions = item[4]    # Process each deck

    for my_deck, my_model, image_items, person_name, subject_name in decks_to_create:

            if processed_regions and processed_regions[0][0] is not None:        # Create notes for the deck

                processed_html = []        for item in image_items:

                original_html = []            original_filename = item[2]

                media_files = []            processed_regions = item[4]



                for idx, orig_crop, proc_crop in processed_regions:            if processed_regions and processed_regions[0][0] is not None:

                    name, ext = os.path.splitext(original_filename)                # For images with blue boxes

                    region_suffix = f"_region{idx+1}"                processed_html = []

                                    original_html = []

                    processed_img_filename = f"{name}{region_suffix}-1_processed{ext}"                media_files = []

                    original_img_filename = f"{name}{region_suffix}-1_original{ext}"

                                    for idx, orig_crop, proc_crop in processed_regions:

                    processed_img_path = os.path.join(output_dir, processed_img_filename)                    name, ext = os.path.splitext(original_filename)

                    original_img_path = os.path.join(output_dir, original_img_filename)                    region_suffix = f"_region{idx+1}"

                    proc_crop.save(processed_img_path)                    

                    orig_crop.save(original_img_path)                    processed_img_filename = f"{name}{region_suffix}-1_processed{ext}"

                                        original_img_filename = f"{name}{region_suffix}-1_original{ext}"

                    original_html.append(f'<img src="{original_img_filename}">')                    

                    processed_html.append(f'<img src="{processed_img_filename}">')                    # Save images

                    media_files.extend([processed_img_path, original_img_path])                    processed_img_path = os.path.join(output_dir, processed_img_filename)

                    original_img_path = os.path.join(output_dir, original_img_filename)

                my_note = genanki.Note(                    proc_crop.save(processed_img_path)

                    model=my_model,                    orig_crop.save(original_img_path)

                    fields=[                    

                        '<br>'.join(original_html),                    original_html.append(f'<img src="{original_img_filename}">')

                        '<br>'.join(processed_html),                    processed_html.append(f'<img src="{processed_img_filename}">')

                        original_filename                    media_files.extend([processed_img_path, original_img_path])

                    ])

                my_deck.add_note(my_note)                my_note = genanki.Note(

                    model=my_model,

            else:                    fields=[

                name, ext = os.path.splitext(original_filename)                        '<br>'.join(original_html),

                processed_img_filename = f"{name}-1_processed{ext}"                        '<br>'.join(processed_html),

                original_img_filename = f"{name}-1_original{ext}"                        original_filename

                                    ])

                processed_img_path = os.path.join(output_dir, processed_img_filename)                my_deck.add_note(my_note)

                original_img_path = os.path.join(output_dir, original_img_filename)

                            else:

                item[4][0][2].save(processed_img_path)                # For images without blue boxes

                item[4][0][1].save(original_img_path)                name, ext = os.path.splitext(original_filename)

                                processed_img_filename = f"{name}-1_processed{ext}"

                my_note = genanki.Note(                original_img_filename = f"{name}-1_original{ext}"

                    model=my_model,                

                    fields=[                processed_img_path = os.path.join(output_dir, processed_img_filename)

                        f'<img src="{original_img_filename}">',                original_img_path = os.path.join(output_dir, original_img_filename)

                        f'<img src="{processed_img_filename}">',                

                        original_filename                item[4][0][2].save(processed_img_path)  # Save processed image

                    ])                item[4][0][1].save(original_img_path)   # Save original image

                my_deck.add_note(my_note)                

                media_files = [processed_img_path, original_img_path]                my_note = genanki.Note(

                    model=my_model,

        base_apkg_filename = f"완성된 {subject_name}.apkg"                    fields=[

        apkg_output_path = os.path.join(anki_output_dir, base_apkg_filename)                        f'<img src="{original_img_filename}">',

                                f'<img src="{processed_img_filename}">',

        counter = 1                        original_filename

        while os.path.exists(apkg_output_path):                    ])

            name, ext = os.path.splitext(base_apkg_filename)                my_deck.add_note(my_note)

            apkg_filename = f"{name}-{counter}{ext}"                media_files = [processed_img_path, original_img_path]

            apkg_output_path = os.path.join(anki_output_dir, apkg_filename)

            counter += 1        # Create and save the Anki package

        base_apkg_filename = f"완성된 {subject_name}.apkg"

        try:        apkg_output_path = os.path.join(anki_output_dir, base_apkg_filename)

            genanki.Package(my_deck, media_files=media_files).write_to_file(apkg_output_path)        

            print(f"안키 패키지 생성 완료: {apkg_output_path}")        # Handle duplicate filenames

        except Exception as e:        counter = 1

            print(f"안키 패키지 생성 오류 ({subject_name}): {e}")        while os.path.exists(apkg_output_path):

            name, ext = os.path.splitext(base_apkg_filename)

    print("\n임시 파일 정리 중...")            apkg_filename = f"{name}-{counter}{ext}"

    if os.path.exists(output_dir):            apkg_output_path = os.path.join(anki_output_dir, apkg_filename)

        try:            counter += 1

            for filename in os.listdir(output_dir):

                file_path = os.path.join(output_dir, filename)        try:

                if os.path.isfile(file_path):            genanki.Package(my_deck, media_files=media_files).write_to_file(apkg_output_path)

                    os.remove(file_path)            print(f"Created Anki package: {apkg_output_path}")

            os.rmdir(output_dir)        except Exception as e:

            print("임시 파일 정리 완료")            print(f"Error creating Anki package for {subject_name}: {e}")

        except Exception as e:

            print(f"임시 파일 정리 중 오류 발생: {e}")    # Clean up

    if os.path.exists(output_dir):

    print("\n모든 처리가 완료되었습니다!")        try:

            for filename in os.listdir(output_dir):

def main():                file_path = os.path.join(output_dir, filename)

    """                if os.path.isfile(file_path):

    명령행 인터페이스를 처리하는 메인 함수                    os.remove(file_path)

    """            os.rmdir(output_dir)

    parser = argparse.ArgumentParser(description='이미지를 처리하여 안키 카드를 생성합니다.')        except Exception as e:

    parser.add_argument('input_dir', nargs='?', default='.',            print(f"Error cleaning up processed_images_output directory: {e}")

                      help='처리할 이미지가 있는 디렉토리 경로 (기본값: 현재 디렉토리)')

    parser.add_argument('--width', type=int, default=1280,    print("Processing completed!")

                      help='이미지 리사이징할 목표 너비 (기본값: 1280)')

if __name__ == "__main__":

    args = parser.parse_args()    process_images()

    try:
        process_images(args.input_dir, args.width)
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()