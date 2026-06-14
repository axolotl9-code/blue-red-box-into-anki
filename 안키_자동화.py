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
import argparse
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
    # parse CLI args (if any)
    parser = argparse.ArgumentParser(description='Anki automation: process images and PDFs into decks')
    parser.add_argument('--import-files', nargs='+', help='Absolute paths to image/pdf files to import (bypass 과목들 traversal)')
    parser.add_argument('--delete-sources', action='store_true', help='If set, delete the original source files processed in this run')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    subjects_dir = os.path.join(base_dir, "과목들")
    anki_cards_dir_name = "완성된 안키 카드들"
    # Use a run-unique output dir so previous runs' files aren't reused
    run_ts = int(time.time())
    output_dir = os.path.join(base_dir, f"processed_images_output_{run_ts}")
    os.makedirs(output_dir, exist_ok=True)

    # image_data_list items: [deck_key, orig_file_name, save_name, image(PIL), processed_regions, source_abs_path]
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

    # If --import-files provided, build image_data_list from those paths
    if args.import_files:
        for file_path in args.import_files:
            if not os.path.isabs(file_path):
                print(f"경로가 절대 경로가 아닙니다. 건너뜁니다: {file_path}")
                continue
            if not os.path.exists(file_path):
                print(f"파일을 찾을 수 없습니다: {file_path}")
                continue

            parent_dir = os.path.basename(os.path.dirname(file_path))
            # For --import-files, use filename stem (without extension) as deck_key for better granularity
            # This avoids all files in Trash folder having the same deck_key
            file_stem = os.path.splitext(os.path.basename(file_path))[0]
            deck_key = file_stem

            if file_path.lower().endswith('.pdf'):
                print(f"PDF 처리 중: {file_path}")
                pdf_images = convert_pdf_to_images(file_path)
                for page_num, pdf_image in enumerate(pdf_images, start=1):
                    pdf_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_page{page_num:03d}.png"
                    image_data_list.append([deck_key, os.path.basename(file_path), pdf_filename, pdf_image, None, file_path])
                    print(f"  페이지 {page_num}/{len(pdf_images)} 추가됨: deck={deck_key}")
            else:
                try:
                    original_image = Image.open(file_path)
                    image_data_list.append([deck_key, os.path.basename(file_path), os.path.basename(file_path), original_image, None, file_path])
                except Exception as e:
                    print(f"이미지 처리 오류: {file_path}, {e}")
    else:
        # Walk the subjects_dir recursively so any folder depth is supported.
        for root, dirs, files in os.walk(subjects_dir):
            # Compute relative path from subjects_dir and use its parts as deck hierarchy
            rel_root = os.path.relpath(root, subjects_dir)
            if rel_root == '.':
                deck_key = ''
            else:
                # Convert file system path to Anki deck path using '::' separator
                parts = [p for p in rel_root.split(os.sep) if p and p != anki_cards_dir_name]
                deck_key = '::'.join(parts)

            # Collect candidate files in this directory
            try:
                candidate_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
            except Exception:
                candidate_files = []

            # Sorting key uses timestamp from file or embedded EXIF; ties broken by filename
            def sort_key(fname):
                path = os.path.join(root, fname)
                ts = get_image_taken_timestamp(path)
                if ts is None:
                    ts_for_sort = float('inf')
                else:
                    ts_for_sort = ts
                return (ts_for_sort, fname.lower())

            candidate_files.sort(key=sort_key)

            for file in candidate_files:
                file_path = os.path.join(root, file)

                # Build the deck identifier as deck_key ('' means top-level subject folder name will be used later)
                # For PDF, convert pages to images and name them with page numbers
                if file.lower().endswith('.pdf'):
                    print(f"PDF 처리 중: {file_path}")
                    pdf_images = convert_pdf_to_images(file_path)
                    for page_num, pdf_image in enumerate(pdf_images, start=1):
                        pdf_filename = f"{os.path.splitext(file)[0]}_page{page_num:03d}.png"
                        image_data_list.append([deck_key, file, pdf_filename, pdf_image, None, file_path])
                        print(f"  페이지 {page_num}/{len(pdf_images)} 추가됨: deck={deck_key}")
                else:
                    try:
                        original_image = Image.open(file_path)
                        image_data_list.append([deck_key, file, file, original_image, None, file_path])
                    except Exception as e:
                        print(f"이미지 처리 오류: {file_path}, {e}")

    # 이미지 리사이즈
    target_width = 1280
    for idx, item in enumerate(image_data_list):
        # item format now: [deck_key, orig_file, save_name, image]
        original_image = item[3]
        original_width, original_height = original_image.size
        if original_width > 0:
            aspect_ratio = original_height / original_width
            new_height = int(target_width * aspect_ratio)
        else:
            new_height = original_height
        try:
            resized_image = original_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
            image_data_list[idx][3] = resized_image
        except Exception as e:
            print(f"리사이즈 오류: {item[2]}, {e}")

    # 이미지 처리 및 저장
    for idx, item in enumerate(image_data_list):
        # item format: [deck_key, orig_file, save_name, image, processed_regions, source_abs_path]
        deck_key, orig_file, save_name, original_image = item[0], item[1], item[2], item[3]
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
        # store processed regions back into the item's slot
        image_data_list[idx][4] = processed_regions

    # 이미지 파일 및 CSV 저장
    csv_output_path = os.path.join(output_dir, "image_pairs.csv")
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for item in image_data_list:
            # item: [deck_key, orig_file, save_name, image, processed_regions]
            save_name = item[2]
            processed_regions = item[4]
            for ridx, orig_crop, proc_crop in processed_regions:
                name, ext = os.path.splitext(save_name)
                region_suffix = f"_region{ridx+1}" if ridx is not None else ""
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
    
    # ✅ 고유한 노트타입 이름 사용 - 절대 충돌 없음
    # 고정된 Model ID (절대 변경하지 말 것!)
    IMAGE_MODEL_ID = 1607392319
    
    # Group images by deck key (deck_key may be '' for top-level folders)
    grouped_data = {}
    for item in image_data_list:
        deck_key = item[0]
        # If deck_key is empty, derive deck name from the source file's parent folder
        if not deck_key:
            source_abs = item[5]
            if source_abs:
                deck_name_key = os.path.basename(os.path.dirname(source_abs))
            else:
                deck_name_key = 'Default'
        else:
            deck_name_key = deck_key
        if deck_name_key not in grouped_data:
            grouped_data[deck_name_key] = []
        grouped_data[deck_name_key].append(item)

    # ✅ 고유한 이름의 노트타입 생성 (Basic 구조 그대로, 이름만 다름)
    my_model = genanki.Model(
        IMAGE_MODEL_ID,
        'ImageBasic',  # 고유한 이름으로 충돌 방지
        fields=[
            {'name': 'Front'},
            {'name': 'Back'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Front}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Back}}',
            },
        ],
        css='.card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }'
    )

    # Collect all decks and their media to write into one .apkg
    all_decks = []  # list of (deck, model, media_files)
    for subject_key, image_items in grouped_data.items():
        # subject_key is the deck hierarchy like '앉아서::물리' (or '' for top-level default)
        if subject_key:
            deck_name = subject_key
        else:
            # fallback: use a generic name if no key (shouldn't normally happen)
            deck_name = 'Default'
        # use deterministic positive deck id from SHA1 to avoid collisions and negative ids
        deck_hash = hashlib.sha1(deck_name.encode('utf-8')).hexdigest()[:12]
        deck_id = int(deck_hash, 16) % (10 ** 12)
        my_deck = genanki.Deck(deck_id, deck_name)
        media_files = []
        for item in image_items:
            # item: [deck_key, orig_file, save_name, image, processed_regions]
            save_name = item[2]
            processed_regions = item[4]
            if processed_regions and processed_regions[0][0] is not None:
                processed_html = []
                original_html = []
                for idx, orig_crop, proc_crop in processed_regions:
                    name, ext = os.path.splitext(save_name)
                    region_suffix = f"_region{idx+1}"
                    processed_img_filename = f"{name}{region_suffix}-1_processed{ext}"
                    original_img_filename = f"{name}{region_suffix}-1_original{ext}"
                    processed_html.append(f'<img src="{processed_img_filename}">')
                    original_html.append(f'<img src="{original_img_filename}">')
                    media_files.append(os.path.join(output_dir, processed_img_filename))
                    media_files.append(os.path.join(output_dir, original_img_filename))
                # Basic 노트타입: Front(문제)와 Back(정답)
                my_note = genanki.Note(
                    model=my_model,
                    fields=[
                        '<br>'.join(processed_html),  # Front: 가려진 이미지
                        '<br>'.join(original_html),   # Back: 원본 이미지
                    ])
                my_deck.add_note(my_note)
            else:
                processed_img_filename = f"{save_name.rsplit('.', 1)[0]}-1_processed.{save_name.rsplit('.', 1)[1]}"
                original_img_filename = f"{save_name.rsplit('.', 1)[0]}-1_original.{save_name.rsplit('.', 1)[1]}"
                media_files.append(os.path.join(output_dir, processed_img_filename))
                media_files.append(os.path.join(output_dir, original_img_filename))
                # Basic 노트타입: Front(문제)와 Back(정답)
                my_note = genanki.Note(
                    model=my_model,
                    fields=[
                        f'<img src="{processed_img_filename}">',  # Front: 가려진 이미지
                        f'<img src="{original_img_filename}">',   # Back: 원본 이미지
                    ])
                my_deck.add_note(my_note)
        base_apkg_filename = f"{deck_name}.apkg"
        apkg_output_path = os.path.join(anki_output_dir, base_apkg_filename)
        counter = 1
        while os.path.exists(apkg_output_path):
            name, ext = os.path.splitext(base_apkg_filename)
            apkg_filename = f"{name}-{counter}{ext}"
            apkg_output_path = os.path.join(anki_output_dir, apkg_filename)
            counter += 1
        # store deck + media for single package creation later
        all_decks.append((my_deck, my_model, media_files, deck_name))

    # Write a single .apkg containing all decks and their media
    combined_name = '완성된_모든_과목.apkg'
    combined_path = os.path.join(anki_output_dir, combined_name)
    counter = 1
    while os.path.exists(combined_path):
        name, ext = os.path.splitext(combined_name)
        combined_name_version = f"{name}-{counter}{ext}"
        combined_path = os.path.join(anki_output_dir, combined_name_version)
        counter += 1

    # collect unique media files across all decks, only include files that actually exist
    combined_media = []
    for _, _, media_files, _ in all_decks:
        for m in media_files:
            if m not in combined_media and os.path.exists(m):
                combined_media.append(m)

    try:
        if not all_decks:
            print("생성할 덱이 없습니다.")
        else:
            decks_only = [t[0] for t in all_decks]
            pkg = genanki.Package(decks_only, media_files=combined_media)
            pkg.write_to_file(combined_path)
            print(f"Anki 통합 패키지 생성 완료: {combined_path}")
    except Exception as e:
        print(f"Anki 패키지 생성 오류: {e}")

    # 원본 이미지 및 PDF 파일 삭제: 이번 실행에서 처리한 파일만 삭제
    processed_source_paths = set()
    for item in image_data_list:
        src = item[5]
        if src:
            processed_source_paths.add(src)

    if args.delete_sources:
        for src_path in processed_source_paths:
            try:
                if os.path.exists(src_path) and os.path.isfile(src_path):
                    os.remove(src_path)
                    print(f"삭제됨: {src_path}")
            except OSError as e:
                print(f"삭제 오류: {src_path}, {e}")
    else:
        if processed_source_paths:
            print("참고: --delete-sources 플래그가 설정되지 않았으므로 원본 파일은 삭제되지 않습니다.")

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
