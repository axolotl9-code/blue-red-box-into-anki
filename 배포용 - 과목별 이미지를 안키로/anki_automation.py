#!/usr/bin/env python3

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
    이미지에서 빨간색 영역을 찾아서 박스 처리합니다.

    Args:
        image_input: 파일 경로(str) 또는 PIL Image 객체

    Returns:
        박스가 그려진 PIL Image 객체
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_input}")
        img_rgb_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_cv2)
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
        img_rgb_cv2 = np.array(img_pil)
        img = cv2.cvtColor(img_rgb_cv2, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("입력은 파일 경로(str) 또는 PIL Image 객체여야 합니다")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 빨간색 범위 정의
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
        if bbox_w > 30 and bbox_h > 5:  # 작은 노이즈 제거
            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # 위에서 아래로, 왼쪽에서 오른쪽으로 정렬

    draw = ImageDraw.Draw(img_pil)
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        draw.rectangle([(x1, y1), (x2, y2)], outline="darkred", fill="darkred", width=2)

    return img_pil

def find_blue_boxes(image_input):
    """
    이미지에서 파란색 영역을 찾아 좌표를 반환합니다.

    Args:
        image_input: 파일 경로(str) 또는 PIL Image 객체

    Returns:
        파란색 박스의 좌표 목록 [(x1, y1, x2, y2), ...], 위에서 아래로 정렬됨
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_input}")
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("입력은 파일 경로(str) 또는 PIL Image 객체여야 합니다")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
        if bbox_w > 30 and bbox_h > 5:  # 작은 노이즈 제거
            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

    boxes = sorted(boxes, key=lambda b: b[1])  # 위에서 아래로 정렬
    return boxes

def crop_image_to_box(image, box):
    """
    이미지를 지정된 박스 영역으로 자릅니다.

    Args:
        image: PIL Image 객체
        box: 좌표 튜플 (x1, y1, x2, y2)

    Returns:
        잘린 영역의 PIL Image 객체
    """
    return image.crop(box)

def process_images(base_dir="."):
    """
    메인 처리 함수: 이미지를 처리하고 안키 카드를 생성합니다.
    
    Args:
        base_dir: 입력 디렉토리 경로 (기본값: 현재 디렉토리)
    """
    print(f"이미지 처리 시작: {base_dir}")
    
    # 기본 설정
    target_width = 1280
    image_data_list = []
    anki_cards_dir_name = "완성된 안키 카드들"

    # 1. 이미지 수집
    try:
        person_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except FileNotFoundError:
        print(f"디렉토리를 찾을 수 없습니다: {base_dir}")
        return

    for person_name in person_dirs:
        person_root = os.path.join(base_dir, person_name)
        
        try:
            subject_dirs = [d for d in os.listdir(person_root) if os.path.isdir(os.path.join(person_root, d))]
        except FileNotFoundError:
            print(f"디렉토리를 찾을 수 없습니다: {person_root}")
            continue

        for subject_name in subject_dirs:
            if subject_name == anki_cards_dir_name:
                continue

            subject_root = os.path.join(person_root, subject_name)

            try:
                for file in os.listdir(subject_root):
                    image_path = os.path.join(subject_root, file)
                    if os.path.isfile(image_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            original_image = Image.open(image_path)
                            original_width, original_height = original_image.size
                            if original_width > 0:
                                aspect_ratio = original_height / original_width
                                new_height = int(target_width * aspect_ratio)
                                resized_image = original_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
                            else:
                                resized_image = original_image
                                print(f"경고: 이미지 너비가 0입니다: {file}")

                            image_data_list.append([
                                person_name,
                                subject_name,
                                os.path.basename(image_path),
                                resized_image
                            ])
                            print(f"이미지 처리 중: {file}")

                        except Exception as e:
                            print(f"이미지 처리 오류 {image_path}: {e}")
            except FileNotFoundError:
                print(f"디렉토리를 찾을 수 없습니다: {subject_root}")
                continue

    if not image_data_list:
        print("처리할 이미지가 없습니다.")
        return

    # 2. 이미지 처리 및 저장
    print("\n파란색/빨간색 영역 처리 중...")
    output_dir = os.path.join(base_dir, "processed_images_output")
    os.makedirs(output_dir, exist_ok=True)

    # 3. 데이터 그룹화
    grouped_data = {}
    for item in image_data_list:
        original_image = item[3]
        try:
            blue_boxes = find_blue_boxes(original_image)
            
            if blue_boxes:
                print(f"{len(blue_boxes)}개의 파란색 영역 발견: {item[2]}")
                processed_regions = []
                for i, blue_box in enumerate(blue_boxes, 1):
                    cropped_image = crop_image_to_box(original_image, blue_box)
                    processed_cropped = boxing(cropped_image)
                    processed_regions.append((i-1, cropped_image, processed_cropped))
                item.append(processed_regions)
            else:
                print(f"전체 이미지 처리 중: {item[2]}")
                processed_image = boxing(original_image)
                item.append([(None, original_image, processed_image)])
        except Exception as e:
            print(f"이미지 처리 오류 {item[2]}: {e}")

        # 그룹화
        person_name = item[0]
        subject_name = item[1]
        if person_name not in grouped_data:
            grouped_data[person_name] = {}
        if subject_name not in grouped_data[person_name]:
            grouped_data[person_name][subject_name] = []
        grouped_data[person_name][subject_name].append(item)

    # 4. 안키 덱 생성
    print("\n안키 덱 생성 중...")
    decks_to_create = []
    base_model_id = 1234567890

    for person_name, subjects_data in grouped_data.items():
        for subject_name, image_items in subjects_data.items():
            print(f"덱 준비 중: {subject_name} ({person_name})")
            subject_hash = int(hashlib.sha256(subject_name.encode('utf-8')).hexdigest(), 16) % (10**9)
            model_id = base_model_id + subject_hash

            my_model = genanki.Model(
                model_id,
                'Simple Image Card',
                # Field order: front (Processed Image), back (Original Image), metadata
                fields=[
                    {'name': 'Processed Image'},
                    {'name': 'Original Image'},
                    {'name': 'Original Filename'},
                ],
                templates=[{
                    'name': 'Card 1',
                    # Front: processed image. Back: original image + filename
                    'qfmt': '{{Processed Image}}',
                    'afmt': '{{FrontSide}}<hr id="answer">{{Original Image}}<br>Original Filename: {{Original Filename}}',
                }])

            my_deck = genanki.Deck(
                abs(hash(subject_name)),
                '자동으로 만든 덱::자동으로 만든 '+subject_name)

            decks_to_create.append((my_deck, my_model, image_items, person_name, subject_name))

    # 5. 안키 패키지 생성
    print("\n안키 패키지 생성 중...")
    anki_output_dir = os.path.join(base_dir, anki_cards_dir_name)
    os.makedirs(anki_output_dir, exist_ok=True)

    for my_deck, my_model, image_items, person_name, subject_name in decks_to_create:
        for item in image_items:
            original_filename = item[2]
            processed_regions = item[4]

            if processed_regions and processed_regions[0][0] is not None:
                processed_html = []
                original_html = []
                media_files = []

                for idx, orig_crop, proc_crop in processed_regions:
                    name, ext = os.path.splitext(original_filename)
                    region_suffix = f"_region{idx+1}"
                    
                    processed_img_filename = f"{name}{region_suffix}-1_processed{ext}"
                    original_img_filename = f"{name}{region_suffix}-1_original{ext}"
                    
                    processed_img_path = os.path.join(output_dir, processed_img_filename)
                    original_img_path = os.path.join(output_dir, original_img_filename)
                    proc_crop.save(processed_img_path)
                    orig_crop.save(original_img_path)
                    
                    processed_html.append(f'<img src="{processed_img_filename}">')
                    original_html.append(f'<img src="{original_img_filename}">')
                    media_files.extend([processed_img_path, original_img_path])

                my_note = genanki.Note(
                    model=my_model,
                    # Field order matches model: Processed Image, Original Image, Original Filename
                    fields=[
                        '<br>'.join(processed_html),
                        '<br>'.join(original_html),
                        original_filename
                    ])
                my_deck.add_note(my_note)

            else:
                name, ext = os.path.splitext(original_filename)
                processed_img_filename = f"{name}-1_processed{ext}"
                original_img_filename = f"{name}-1_original{ext}"
                
                processed_img_path = os.path.join(output_dir, processed_img_filename)
                original_img_path = os.path.join(output_dir, original_img_filename)
                
                item[4][0][2].save(processed_img_path)
                item[4][0][1].save(original_img_path)
                
                my_note = genanki.Note(
                    model=my_model,
                    # Field order matches model: Processed Image (front), Original Image (back), Filename
                    fields=[
                        f'<img src="{processed_img_filename}">',
                        f'<img src="{original_img_filename}">',
                        original_filename
                    ])
                my_deck.add_note(my_note)
                media_files = [processed_img_path, original_img_path]

            try:
                base_apkg_filename = f"완성된 {subject_name}.apkg"
                apkg_output_path = os.path.join(anki_output_dir, base_apkg_filename)
                
                counter = 1
                while os.path.exists(apkg_output_path):
                    name, ext = os.path.splitext(base_apkg_filename)
                    apkg_filename = f"{name}-{counter}{ext}"
                    apkg_output_path = os.path.join(anki_output_dir, apkg_filename)
                    counter += 1

                genanki.Package(my_deck, media_files=media_files).write_to_file(apkg_output_path)
                print(f"안키 패키지 생성 완료: {apkg_output_path}")
            except Exception as e:
                print(f"안키 패키지 생성 오류 ({subject_name}): {e}")

    # 6. 임시 파일 정리
    print("\n임시 파일 정리 중...")
    if os.path.exists(output_dir):
        try:
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(output_dir)
            print("임시 파일 정리 완료")
        except Exception as e:
            print(f"임시 파일 정리 중 오류 발생: {e}")

    # 7. 처리 완료된 원본 이미지 파일 삭제
    print("\n처리 완료된 원본 이미지 삭제 중...")
    for root, dirs, files in os.walk(base_dir, topdown=True):
        # 안키 카드 디렉토리는 제외
        if anki_cards_dir_name in dirs:
            dirs.remove(anki_cards_dir_name)

        for file in files:
            # 이미지 파일만 처리
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"삭제됨: {file_path}")
                except OSError as e:
                    print(f"파일 삭제 중 오류 발생 {file_path}: {e}")

    print("\n모든 처리가 완료되었습니다!")

if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")