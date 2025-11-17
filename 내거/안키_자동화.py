# 필요한 패키지 설치 안내
# pip install genanki pandas pillow opencv-python numpy pymupdf

import genanki
import hashlib
import os
import csv
import shutil
import io
from PIL import Image, ImageDraw
import cv2
import numpy as np
import datetime
import time
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PyMuPDF가 설치되지 않았습니다. PDF 지원을 위해 'pip install pymupdf'를 실행하세요.")

def format_deck_name(subject_name):
    """
    과목 이름에 괄호가 있을 경우,
    괄호 안의 내용을 상위 덱 이름으로,
    괄호 밖의 내용을 하위 덱 이름으로 변환한다.
    예: '사회학의 이해(앉아서)' -> '앉아서::사회학의 이해'
    """
    if "(" not in subject_name or ")" not in subject_name:
        return subject_name
    # 마지막 괄호 쌍 기준으로 분리
    open_idx = subject_name.rfind("(")
    close_idx = subject_name.rfind(")")
    if open_idx == -1 or close_idx == -1 or open_idx > close_idx:
        return subject_name
    parent = subject_name[open_idx + 1:close_idx].strip()
    child = (subject_name[:open_idx] + subject_name[close_idx + 1:]).strip()
    if not parent or not child:
        return subject_name
    return f"{parent}::{child}"

def boxing(image_input):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_input}")
        img_rgb_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_cv2)
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
        img_rgb_cv2 = np.array(img_pil)
        img = cv2.cvtColor(img_rgb_cv2, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("Input must be a file path (str) or a PIL Image object")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
        if bbox_w > 30 and bbox_h > 5:
            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    draw = ImageDraw.Draw(img_pil)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([(int(x1), int(y1)), (int(x2), int(y2))], outline="darkred", fill="darkred", width=2)
    return img_pil

def find_blue_boxes(image_input):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_input}")
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("Input must be a file path (str) or a PIL Image object")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
        if bbox_w > 30 and bbox_h > 5:
            boxes.append((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))
    boxes = sorted(boxes, key=lambda b: b[1])
    return boxes

def crop_image_to_box(image, box):
    return image.crop(box)

def convert_pdf_to_images(pdf_path):
    """
    PDF 파일을 이미지 리스트로 변환합니다.
    각 페이지가 하나의 이미지가 됩니다.
    
    Returns:
        List of PIL Image objects, one per page in order
    """
    if not PDF_SUPPORT:
        print(f"PDF 지원이 비활성화되어 있습니다: {pdf_path}")
        return []
    
    try:
        # PyMuPDF를 사용하여 PDF를 이미지로 변환
        pdf_document = fitz.open(pdf_path)
        images = []
        
        # 각 페이지를 고해상도 이미지로 변환
        zoom = 2.0  # 해상도 증가 (2배)
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=mat)
            
            # Pixmap을 PIL Image로 변환
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
        return images
    except Exception as e:
        print(f"PDF 변환 오류: {pdf_path}, {e}")
        return []

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    subjects_dir = os.path.join(base_dir, "과목들")
    anki_cards_dir_name = "완성된 안키 카드들"
    output_dir = os.path.join(base_dir, "processed_images_output")
    os.makedirs(output_dir, exist_ok=True)

    image_data_list = []
    # 과목 폴더 탐색
    if not os.path.exists(subjects_dir):
        print(f"과목들 폴더가 없습니다: {subjects_dir}")
        return

    def get_image_taken_timestamp(path):
        """Return image taken time as a POSIX timestamp (float) if available from EXIF DateTimeOriginal.
        Fallback to file modification time. Return None only if both fail.
        """
        try:
            with Image.open(path) as im:
                exif = im.getexif()
                if exif:
                    # 36867 is the tag for DateTimeOriginal
                    dto = exif.get(36867) or exif.get(306)  # try DateTimeOriginal then DateTime
                    if dto:
                        # EXIF datetime format: 'YYYY:MM:DD HH:MM:SS'
                        try:
                            dt = datetime.datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
                            return dt.timestamp()
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            return os.path.getmtime(path)
        except Exception:
            return None

    for subject_name in os.listdir(subjects_dir):
        subject_path = os.path.join(subjects_dir, subject_name)
        if not os.path.isdir(subject_path):
            continue
        # Collect image files and PDF files
        try:
            candidate_files = [f for f in os.listdir(subject_path)
                               if os.path.isfile(os.path.join(subject_path, f))
                               and f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        except Exception:
            candidate_files = []

        # Sort by photo taken time (EXIF DateTimeOriginal) if available.
        # Fallback: file modification time, then filename to ensure deterministic order.
        def sort_key(fname):
            path = os.path.join(subject_path, fname)
            ts = get_image_taken_timestamp(path)
            # None should sort after real timestamps; use a tuple so filename acts as tiebreaker
            if ts is None:
                ts_for_sort = float('inf')
            else:
                ts_for_sort = ts
            return (ts_for_sort, fname.lower())

        # 오래된 파일이 먼저 오도록 정렬 (순서대로 카드가 만들어지도록)
        candidate_files.sort(key=sort_key)

        for file in candidate_files:
            file_path = os.path.join(subject_path, file)
            
            # PDF 파일 처리
            if file.lower().endswith('.pdf'):
                print(f"PDF 처리 중: {file}")
                pdf_images = convert_pdf_to_images(file_path)
                for page_num, pdf_image in enumerate(pdf_images, start=1):
                    # PDF의 각 페이지를 별도 이미지로 처리
                    # 파일명에 페이지 번호 추가
                    pdf_filename = f"{os.path.splitext(file)[0]}_page{page_num:03d}.png"
                    image_data_list.append([subject_name, pdf_filename, pdf_image])
                    print(f"  페이지 {page_num}/{len(pdf_images)} 추가됨")
            else:
                # 일반 이미지 파일 처리
                try:
                    original_image = Image.open(file_path)
                    image_data_list.append([subject_name, file, original_image])
                except Exception as e:
                    print(f"이미지 처리 오류: {file_path}, {e}")

    # 이미지 리사이즈
    target_width = 1280
    for item in image_data_list:
        original_image = item[2]
        original_width, original_height = original_image.size
        if original_width > 0:
            aspect_ratio = original_height / original_width
            new_height = int(target_width * aspect_ratio)
        else:
            new_height = original_height
        try:
            resized_image = original_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
            item[2] = resized_image
        except Exception as e:
            print(f"리사이즈 오류: {item[1]}, {e}")

    # 이미지 처리 및 저장
    for item in image_data_list:
        subject_name, original_filename, original_image = item
        blue_boxes = find_blue_boxes(original_image)
        processed_regions = []
        if blue_boxes:
            for i, blue_box in enumerate(blue_boxes):
                cropped_image = crop_image_to_box(original_image, blue_box)
                processed_cropped = boxing(cropped_image)
                processed_regions.append((i, cropped_image, processed_cropped))
        else:
            processed_image = boxing(original_image)
            processed_regions.append((None, original_image, processed_image))
        item.append(processed_regions)

    # 이미지 파일 및 CSV 저장
    csv_output_path = os.path.join(output_dir, "image_pairs.csv")
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for item in image_data_list:
            original_filename = item[1]
            processed_regions = item[3]
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

    # 안키 덱 생성
    anki_output_dir = os.path.join(base_dir, anki_cards_dir_name)
    os.makedirs(anki_output_dir, exist_ok=True)
    base_model_id = 1234567890
    grouped_data = {}
    for item in image_data_list:
        subject_name = item[0]
        if subject_name not in grouped_data:
            grouped_data[subject_name] = []
        grouped_data[subject_name].append(item)

    for subject_name, image_items in grouped_data.items():
        subject_hash = int(hashlib.sha256(subject_name.encode('utf-8')).hexdigest(), 16) % (10**9)
        model_id = base_model_id + subject_hash
        deck_name = format_deck_name(subject_name)
        my_model = genanki.Model(
            model_id,
            'Simple Image Card',
            fields=[
                {'name': 'Processed Image'},
                {'name': 'Original Image'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '{{Processed Image}}',
                    'afmt': '{{FrontSide}}<hr id="answer">{{Original Image}}',
                },
            ])
        my_deck = genanki.Deck(
            abs(hash(subject_name)), deck_name)
        media_files = []
        for item in image_items:
            original_filename = item[1]
            processed_regions = item[3]
            if processed_regions and processed_regions[0][0] is not None:
                processed_html = []
                original_html = []
                for idx, orig_crop, proc_crop in processed_regions:
                    name, ext = os.path.splitext(original_filename)
                    region_suffix = f"_region{idx+1}"
                    processed_img_filename = f"{name}{region_suffix}-1_processed{ext}"
                    original_img_filename = f"{name}{region_suffix}-1_original{ext}"
                    processed_html.append(f'<img src="{processed_img_filename}">')
                    original_html.append(f'<img src="{original_img_filename}">')
                    media_files.append(os.path.join(output_dir, processed_img_filename))
                    media_files.append(os.path.join(output_dir, original_img_filename))
                my_note = genanki.Note(
                    model=my_model,
                    fields=[
                        '<br>'.join(processed_html),
                        '<br>'.join(original_html),
                    ])
                my_deck.add_note(my_note)
            else:
                processed_img_filename = f"{original_filename.rsplit('.', 1)[0]}-1_processed.{original_filename.rsplit('.', 1)[1]}"
                original_img_filename = f"{original_filename.rsplit('.', 1)[0]}-1_original.{original_filename.rsplit('.', 1)[1]}"
                media_files.append(os.path.join(output_dir, processed_img_filename))
                media_files.append(os.path.join(output_dir, original_img_filename))
                my_note = genanki.Note(
                    model=my_model,
                    fields=[
                        f'<img src="{processed_img_filename}">',
                        f'<img src="{original_img_filename}">',
                    ])
                my_deck.add_note(my_note)
        base_apkg_filename = f"{subject_name}.apkg"
        apkg_output_path = os.path.join(anki_output_dir, base_apkg_filename)
        counter = 1
        while os.path.exists(apkg_output_path):
            name, ext = os.path.splitext(base_apkg_filename)
            apkg_filename = f"{name}-{counter}{ext}"
            apkg_output_path = os.path.join(anki_output_dir, apkg_filename)
            counter += 1
        try:
            genanki.Package(my_deck, media_files=media_files).write_to_file(apkg_output_path)
            print(f"Anki 패키지 생성 완료: {apkg_output_path}")
        except Exception as e:
            print(f"Anki 패키지 생성 오류: {e}")

    # 원본 이미지 및 PDF 파일 삭제
    for subject_name in os.listdir(subjects_dir):
        subject_path = os.path.join(subjects_dir, subject_name)
        if not os.path.isdir(subject_path):
            continue
        for file in os.listdir(subject_path):
            file_path = os.path.join(subject_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                try:
                    os.remove(file_path)
                    print(f"삭제됨: {file_path}")
                except OSError as e:
                    print(f"삭제 오류: {file_path}, {e}")

    # processed_images_output 폴더 정리
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"삭제됨: {file_path}")
        try:
            os.rmdir(output_dir)
            print(f"폴더 삭제됨: {output_dir}")
        except Exception as e:
            print(f"폴더 삭제 오류: {e}")

if __name__ == "__main__":
    main()
