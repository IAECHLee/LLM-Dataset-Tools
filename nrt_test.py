"""
NRT 권취 줄(Coil Trace) 위치 추적 테스트 스크립트
"""
import nrt
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json

# 경로 설정
MODEL_PATH = r"D:\LLM_Dataset\models\Trace_Coil.net"
IMAGE_DIR = r"K:\LLM Image_Storage\A line-2025-07-25_09-49-08(정상)"
OUTPUT_DIR = r"D:\LLM_Dataset\tracking_results"
INPUT_SIZE = 512  # 모델 입력 크기

def imread_korean(path):
    """한글 경로 이미지 로드"""
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

def preprocess_image(img, target_size=512):
    """이미지 전처리: 512x512 리사이즈"""
    return cv2.resize(img, (target_size, target_size))

def test_nrt_model():
    print("=" * 60)
    print("NRT Coil Trace Position Tracking Test")
    print("=" * 60)
    
    # 1. 모델 로드
    print(f"\n[1] Loading model: {MODEL_PATH}")
    try:
        predictor = nrt.Predictor(MODEL_PATH)
        print("    ✓ Model loaded successfully!")
    except Exception as e:
        print(f"    ✗ Failed to load model: {e}")
        return
    
    # 2. 모델 정보 출력
    print("\n[2] Model Info:")
    print(f"    - Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    
    # 3. 테스트 이미지 로드 (전체)
    print(f"\n[3] Loading test images from: {IMAGE_DIR}")
    
    image_dir = Path(IMAGE_DIR)
    image_files = sorted([f.name for f in image_dir.glob('*.jpg')])
    total_images = len(image_files)
    print(f"    - Total images: {total_images}")
    print(f"    - Processing all images...")
    
    # 위치 추적 결과 저장
    tracking_results = []
    
    for idx, img_file in enumerate(image_files):
        img_path = image_dir / img_file
        img = imread_korean(str(img_path))
        
        # 진행률 표시 (100장마다)
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"    Processing: {idx+1}/{total_images} ({(idx+1)/total_images*100:.1f}%)")
        
        if img is None:
            print(f"    ✗ Failed to load: {img_file}")
            continue
        
        orig_h, orig_w = img.shape[:2]
        
        # 4. 전처리: 512x512 리사이즈
        img_resized = preprocess_image(img, INPUT_SIZE)
        
        # 5. 추론 실행
        try:
            # nrt.Input 형식으로 변환
            input_data = nrt.Input()
            image_buff = nrt.NDBuffer.from_numpy(img_resized)
            input_data.extend(image_buff)
            
            result = predictor.predict(input_data)
            input_data.clear()
            
            # 결과 출력 - Detection 결과 (권취 줄 위치)
            if hasattr(result, 'bboxes'):
                bbox_count = result.bboxes.get_count()
                
                if bbox_count > 0:
                    # 첫 번째 bbox 선택 (또는 가장 큰 것)
                    best_bbox = None
                    best_area = 0
                    for i in range(bbox_count):
                        bbox = result.bboxes.get(i)
                        area = bbox.rect.width * bbox.rect.height
                        if area > best_area:
                            best_area = area
                            best_bbox = bbox
                    
                    if best_bbox:
                        # 512x512 좌표를 원본 좌표로 변환
                        scale_x = orig_w / INPUT_SIZE
                        scale_y = orig_h / INPUT_SIZE
                        
                        orig_x = int(best_bbox.rect.x * scale_x)
                        orig_y = int(best_bbox.rect.y * scale_y)
                        orig_w_box = int(best_bbox.rect.width * scale_x)
                        orig_h_box = int(best_bbox.rect.height * scale_y)
                        center_x = orig_x + orig_w_box // 2
                        center_y = orig_y + orig_h_box // 2
                        
                        tracking_results.append({
                            'frame': idx,
                            'file': img_file,
                            'center': (center_x, center_y),
                            'bbox': (orig_x, orig_y, orig_w_box, orig_h_box),
                            'class_idx': best_bbox.class_idx
                        })
                        
                        # 탐지 시 출력 (처음 10개, 이후 100개마다)
                        if len(tracking_results) <= 10 or len(tracking_results) % 100 == 0:
                            print(f"    [{idx:5d}] {img_file}: Center=({center_x:4d}, {center_y:4d}), Class={best_bbox.class_idx}")
            else:
                pass  # No bbox result - skip silently
                
        except Exception as e:
            print(f"    ✗ Prediction failed: {e}")
    
    # 6. 추적 결과 요약
    print("\n" + "=" * 60)
    print("TRACKING SUMMARY")
    print("=" * 60)
    print(f"Total frames tested: {len(image_files)}")
    print(f"Frames with coil detected: {len(tracking_results)}")
    
    if tracking_results:
        centers = [r['center'] for r in tracking_results]
        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]
        print(f"X range: {min(x_coords)} ~ {max(x_coords)}")
        print(f"Y range: {min(y_coords)} ~ {max(y_coords)}")
    
    print("=" * 60)
    
    # 7. 결과 저장
    save_tracking_results(tracking_results, image_files)

def save_tracking_results(tracking_results, image_files):
    """추적 결과를 파일로 저장"""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. TXT 파일 저장 (간단한 형식)
    txt_file = output_path / f"tracking_{timestamp}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("# Coil Trace Tracking Results\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Source: {IMAGE_DIR}\n")
        f.write(f"# Total frames: {len(image_files)}, Detected: {len(tracking_results)}\n")
        f.write("#\n")
        f.write("# Format: frame_num, filename, center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, class_idx\n")
        f.write("#" + "=" * 80 + "\n")
        
        for r in tracking_results:
            f.write(f"{r['frame']}, {r['file']}, {r['center'][0]}, {r['center'][1]}, ")
            f.write(f"{r['bbox'][0]}, {r['bbox'][1]}, {r['bbox'][2]}, {r['bbox'][3]}, {r['class_idx']}\n")
    
    print(f"\n[SAVED] TXT file: {txt_file}")
    
    # 2. CSV 파일 저장 (분석용)
    csv_file = output_path / f"tracking_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("frame,filename,center_x,center_y,bbox_x,bbox_y,bbox_w,bbox_h,class_idx\n")
        for r in tracking_results:
            f.write(f"{r['frame']},{r['file']},{r['center'][0]},{r['center'][1]},")
            f.write(f"{r['bbox'][0]},{r['bbox'][1]},{r['bbox'][2]},{r['bbox'][3]},{r['class_idx']}\n")
    
    print(f"[SAVED] CSV file: {csv_file}")
    
    # 3. JSON 파일 저장 (상세 정보)
    json_file = output_path / f"tracking_{timestamp}.json"
    json_data = {
        'metadata': {
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_dir': IMAGE_DIR,
            'model_path': MODEL_PATH,
            'input_size': INPUT_SIZE,
            'total_frames': len(image_files),
            'detected_frames': len(tracking_results)
        },
        'tracking_data': tracking_results
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVED] JSON file: {json_file}")
    
    return txt_file, csv_file, json_file

if __name__ == "__main__":
    test_nrt_model()
