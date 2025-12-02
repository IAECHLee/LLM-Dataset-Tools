#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""샘플링 테스트 스크립트"""

from pathlib import Path
import layer_sampler as ls

# 설정
root = Path(r"K:\LLM Image_Storage")
keywords = ["A line", "2025-07-27"]
layers_file = Path("layers.json")

print("=" * 60)
print("샘플링 테스트")
print("=" * 60)

# 1. 루트 폴더 확인
print(f"\n1. 루트 폴더 확인")
print(f"   경로: {root}")
print(f"   존재: {root.exists()}")

if not root.exists():
    print("   [ERROR] 루트 폴더가 존재하지 않습니다!")
    exit(1)

# 2. 키워드로 폴더 찾기
print(f"\n2. 키워드로 폴더 찾기")
print(f"   키워드: {keywords}")

target_folders = ls.find_target_folders(root, keywords)
print(f"   발견된 폴더 수: {len(target_folders)}")

if not target_folders:
    print("   [ERROR] 키워드를 포함하는 폴더를 찾을 수 없습니다!")
    print(f"\n   첫 10개 폴더:")
    for folder in list(root.iterdir())[:10]:
        if folder.is_dir():
            print(f"      - {folder.name}")
    exit(1)

for i, folder in enumerate(target_folders[:5], 1):
    print(f"   {i}. {folder.name}")

# 3. 레이어 파일 확인
print(f"\n3. 레이어 파일 확인")
print(f"   경로: {layers_file}")
print(f"   존재: {layers_file.exists()}")

if not layers_file.exists():
    print("   [ERROR] 레이어 파일이 존재하지 않습니다!")
    exit(1)

import json
with open(layers_file, 'r', encoding='utf-8') as f:
    layer_data = json.load(f)

first_winding_id = list(layer_data.keys())[0]
layers = [[layer['start'], layer['end'], layer['count']] for layer in layer_data[first_winding_id]]
print(f"   Winding ID: {first_winding_id}")
print(f"   레이어 수: {len(layers)}")
print(f"   첫 3개 레이어: {layers[:3]}")

# 4. 첫 번째 폴더 처리 테스트
print(f"\n4. 첫 번째 폴더 처리 테스트")
test_folder = target_folders[0]
print(f"   폴더: {test_folder.name}")
print(f"   정상 폴더: {'(정상)' in test_folder.name}")
print(f"   불량 폴더: {'(불량)' in test_folder.name}")

# 이미지 개수 확인
images = ls.list_images(test_folder)
print(f"   이미지 수: {len(images)}")

if len(images) > 0:
    print(f"   첫 이미지: {images[0].name}")
    print(f"   마지막 이미지: {images[-1].name}")
else:
    print("   [ERROR] 이미지가 없습니다!")
    exit(1)

# 5. 샘플링 테스트
print(f"\n5. 샘플링 테스트 (5%)")
sample_ratio = 0.05
sampled = ls.process_folder_with_layers(test_folder, layers, sample_ratio, seed=42)

print(f"   Normal 레이어 수: {len(sampled['normal'])}")
print(f"   Defect 레이어 수: {len(sampled['defect'])}")

total_normal = sum(len(imgs) for imgs in sampled['normal'].values())
total_defect = sum(len(imgs) for imgs in sampled['defect'].values())

print(f"   Normal 이미지 수: {total_normal}")
print(f"   Defect 이미지 수: {total_defect}")

if total_normal > 0:
    print(f"\n   Normal 레이어별 이미지 수:")
    for layer_id, imgs in sorted(sampled['normal'].items()):
        print(f"      Layer {layer_id:02d}: {len(imgs)}개")

if total_defect > 0:
    print(f"\n   Defect 레이어별 이미지 수:")
    for layer_id, imgs in sorted(sampled['defect'].items()):
        print(f"      Layer {layer_id:02d}: {len(imgs)}개")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
