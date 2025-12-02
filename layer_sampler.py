
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-aware dataset sampler for coiling images.
K:\LLM Image_Storage 폴더에서 A line, 2025-07-27이 포함된 폴더를 찾아서
(정상), (불량) 이미지를 층별로 5% 샘플링하여 Normal Layer, Defect Layer로 복사
"""
import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def read_layers_json(json_path: Path, winding_id: str = None) -> List[Tuple[int,int,int]]:
    """
    JSON 파일에서 레이어 범위를 읽음
    형식: {"winding_id": [{"layer": 1, "start": 23, "end": 153, "count": 130}, ...]}
    
    Args:
        json_path: JSON 파일 경로
        winding_id: 특정 winding ID (없으면 첫 번째 winding 사용)
    
    Returns:
        List of (start, end, count) tuples
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # winding_id가 지정되지 않으면 첫 번째 winding 사용
    if winding_id is None:
        winding_id = list(data.keys())[0]
    
    if winding_id not in data:
        raise ValueError(f"winding_id '{winding_id}' not found in JSON. Available: {list(data.keys())}")
    
    layers = data[winding_id]
    rows = [[layer['start'], layer['end'], layer['count']] for layer in layers]
    rows.sort(key=lambda r: r[0])
    return rows



def assign_layers(images: List[Path], layers: List[Tuple[int,int,int]]) -> Dict[int, List[Tuple[int,Path]]]:
    mapping = {i+1: [] for i in range(len(layers))}
    bounds = [(s,e) for (s,e,_) in layers]
    for idx, img in enumerate(images, start=1):
        for i,(s,e) in enumerate(bounds, start=1):
            if s <= idx <= e:
                mapping[i].append((idx, img))
                break
    return mapping

def ensure_dir(p: Path):
    """디렉토리 생성"""
    p.mkdir(parents=True, exist_ok=True)

def find_target_folders(root: Path, keywords: List[str]) -> List[Path]:
    """
    루트 폴더에서 키워드를 모두 포함하는 폴더 찾기
    """
    target_folders = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        folder_name = folder.name
        if all(keyword in folder_name for keyword in keywords):
            target_folders.append(folder)
    return sorted(target_folders, key=lambda p: natural_key(p.name))

def process_folder_with_layers(folder: Path, layers: List[Tuple[int,int,int]], 
                                sample_ratio: float, seed: int, 
                                exclude_images: set = None) -> Dict[str, Dict[int, List[Path]]]:
    """
    폴더에서 이미지를 층별로 샘플링
    폴더명에 (정상) 또는 (불량)이 포함되어 있음
    
    Args:
        folder: 샘플링할 폴더
        layers: 레이어 정의 리스트
        sample_ratio: 샘플링 비율 (0.0 ~ 1.0)
        seed: 랜덤 시드
        exclude_images: 제외할 이미지 파일명 집합 (중복 방지용)
    
    Returns:
        {"normal": {layer_id: [image_paths]}, "defect": {layer_id: [image_paths]}}
    """
    result = {"normal": {}, "defect": {}}
    rng = random.Random(seed)
    
    if exclude_images is None:
        exclude_images = set()
    
    # 폴더명으로 정상/불량 판단
    is_normal = "(정상)" in folder.name
    is_defect = "(불량)" in folder.name
    
    if not is_normal and not is_defect:
        return result
    
    category = "normal" if is_normal else "defect"
    
    # 폴더 내 이미지 직접 읽기
    images = list_images(folder)
    if not images:
        return result
    
    # 제외 목록에 있는 이미지 필터링
    images = [img for img in images if img.name not in exclude_images]
    if not images:
        return result
    
    # 이미지를 층별로 분류
    layer_map = assign_layers(images, layers)
    
    # 각 층에서 sample_ratio 비율만큼 샘플링
    for layer_id, items in layer_map.items():
        if not items:
            continue
            
        # 샘플링할 개수 계산 (최소 1개)
        sample_count = max(1, int(len(items) * sample_ratio))
        sampled = rng.sample(items, min(sample_count, len(items)))
        
        if layer_id not in result[category]:
            result[category][layer_id] = []
        result[category][layer_id].extend([path for _, path in sampled])
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="A line 20250727 이미지 층별 샘플링 프로그램",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python layer_sampler.py --root "K:\LLM Image_Storage" --out "D:\LLM_Dataset\output" --layers-file layers.json --sample-ratio 0.05
  
설명:
  - K:\LLM Image_Storage 폴더에서 "A line"과 "20250727"이 포함된 폴더를 찾습니다
  - 각 폴더 내의 (정상), (불량) 하위 폴더에서 이미지를 처리합니다
  - layers.json의 19개 층 정보에 맞춰 이미지를 분류합니다
  - 각 층에서 지정된 비율(기본 5%)만큼 랜덤 샘플링합니다
  - Normal Layer와 Defect Layer 폴더로 복사합니다
        """
    )
    parser.add_argument("--root", default=r"K:\LLM Image_Storage", help="이미지가 있는 루트 디렉토리 (기본: K:\LLM Image_Storage)")
    parser.add_argument("--out", default=r"D:\LLM_Dataset\output", help="출력 디렉토리 (기본: D:\LLM_Dataset\output)")
    parser.add_argument("--layers-file", type=Path, default=Path("layers.json"), help="레이어 범위 JSON 파일 (기본: layers.json)")
    parser.add_argument("--sample-ratio", type=float, default=0.05, help="각 층에서 샘플링할 비율 (기본: 0.05 = 5%%)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본: 42)")
    parser.add_argument("--keywords", nargs="+", default=["A line", "2025-07-27"], help="검색할 키워드 (기본: A line 2025-07-27)")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    
    # 출력 폴더 구조: Normal Layer, Defect Layer
    normal_dir = out / "Normal Layer"
    defect_dir = out / "Defect Layer"
    ensure_dir(normal_dir)
    ensure_dir(defect_dir)
    
    # 레이어 파일 읽기
    if not args.layers_file.exists():
        print(f"[ERROR] 레이어 파일을 찾을 수 없습니다: {args.layers_file}")
        return
    
    # JSON 파일에서 첫 번째 winding ID 사용
    with open(args.layers_file, 'r', encoding='utf-8') as f:
        layer_data = json.load(f)
    
    # 첫 번째 winding의 레이어 정보 사용
    first_winding_id = list(layer_data.keys())[0]
    layers = [[layer['start'], layer['end'], layer['count']] for layer in layer_data[first_winding_id]]
    num_layers = len(layers)
    
    print(f"[INFO] 레이어 정보: {num_layers}개 층 ({first_winding_id})")
    print(f"[INFO] 루트 디렉토리: {root}")
    print(f"[INFO] 검색 키워드: {args.keywords}")
    print(f"[INFO] 샘플링 비율: {args.sample_ratio * 100:.1f}%")
    
    # 루트 폴더 확인
    if not root.exists():
        print(f"[ERROR] 루트 디렉토리를 찾을 수 없습니다: {root}")
        return
    
    # A line, 20250727이 포함된 폴더 찾기
    target_folders = find_target_folders(root, args.keywords)
    
    if not target_folders:
        print(f"[WARN] 키워드 {args.keywords}를 모두 포함하는 폴더를 찾을 수 없습니다")
        return
    
    print(f"\n[INFO] 발견된 폴더: {len(target_folders)}개")
    for folder in target_folders:
        print(f"  - {folder.name}")
    
    # 통계 정보
    total_stats = {
        "normal": {layer_id: 0 for layer_id in range(1, num_layers + 1)},
        "defect": {layer_id: 0 for layer_id in range(1, num_layers + 1)}
    }
    manifest = []
    
    # 각 폴더 처리 (각 폴더마다 다른 시드 사용)
    for folder_idx, folder in enumerate(target_folders):
        print(f"\n[INFO] 처리 중: {folder.name}")
        
        # 층별로 이미지 샘플링 (폴더마다 완전히 다른 시드)
        folder_hash = hash(folder.name) % 10000
        folder_seed = args.seed + folder_hash + (folder_idx * 1000)
        sampled_images = process_folder_with_layers(folder, layers, args.sample_ratio, folder_seed)
        
        # Normal 이미지 복사
        for layer_id, image_paths in sampled_images["normal"].items():
            if not image_paths:
                continue
            layer_folder = normal_dir / f"Layer_{layer_id:02d}"
            ensure_dir(layer_folder)
            
            for img_path in image_paths:
                dst_name = f"{folder.name}_L{layer_id:02d}_{img_path.name}"
                dst_path = layer_folder / dst_name
                shutil.copy2(img_path, dst_path)
                manifest.append({
                    "source": str(img_path),
                    "destination": str(dst_path),
                    "folder": folder.name,
                    "category": "normal",
                    "layer": layer_id,
                    "filename": img_path.name
                })
                total_stats["normal"][layer_id] += 1
            
            print(f"  - Normal Layer {layer_id:02d}: {len(image_paths)}개 복사")
        
        # Defect 이미지 복사
        for layer_id, image_paths in sampled_images["defect"].items():
            if not image_paths:
                continue
            layer_folder = defect_dir / f"Layer_{layer_id:02d}"
            ensure_dir(layer_folder)
            
            for img_path in image_paths:
                dst_name = f"{folder.name}_L{layer_id:02d}_{img_path.name}"
                dst_path = layer_folder / dst_name
                shutil.copy2(img_path, dst_path)
                manifest.append({
                    "source": str(img_path),
                    "destination": str(dst_path),
                    "folder": folder.name,
                    "category": "defect",
                    "layer": layer_id,
                    "filename": img_path.name
                })
                total_stats["defect"][layer_id] += 1
            
            print(f"  - Defect Layer {layer_id:02d}: {len(image_paths)}개 복사")
    
    # Manifest 저장
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    # 통계 저장
    stats_path = out / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(total_stats, f, indent=2, ensure_ascii=False)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"[완료] 이미지 샘플링 완료")
    print(f"{'='*60}")
    print(f"\n정상 이미지 (Normal Layer):")
    total_normal = 0
    for layer_id in range(1, num_layers + 1):
        count = total_stats["normal"][layer_id]
        if count > 0:
            print(f"  Layer {layer_id:02d}: {count}개")
            total_normal += count
    print(f"  합계: {total_normal}개")
    
    print(f"\n불량 이미지 (Defect Layer):")
    total_defect = 0
    for layer_id in range(1, num_layers + 1):
        count = total_stats["defect"][layer_id]
        if count > 0:
            print(f"  Layer {layer_id:02d}: {count}개")
            total_defect += count
    print(f"  합계: {total_defect}개")
    
    print(f"\n출력 폴더: {out}")
    print(f"  - Normal Layer: {normal_dir}")
    print(f"  - Defect Layer: {defect_dir}")
    print(f"  - Manifest: {manifest_path}")
    print(f"  - 통계: {stats_path}")
    print(f"\n총 {total_normal + total_defect}개 이미지 복사 완료")

if __name__ == "__main__":
    main()
