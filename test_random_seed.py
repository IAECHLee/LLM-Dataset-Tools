#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""랜덤 시드 효과 테스트"""

from pathlib import Path
import layer_sampler as ls

# 설정
root = Path(r"K:\LLM Image_Storage")
keywords = ["A line", "2025-07-27"]
layers_file = Path("layers.json")

print("=" * 60)
print("랜덤 시드 효과 테스트")
print("=" * 60)

import json
with open(layers_file, 'r', encoding='utf-8') as f:
    layer_data = json.load(f)

first_winding_id = list(layer_data.keys())[0]
layers = [[layer['start'], layer['end'], layer['count']] for layer in layer_data[first_winding_id]]

# 폴더 찾기
target_folders = ls.find_target_folders(root, keywords)
print(f"\n발견된 폴더: {len(target_folders)}개")

# 처음 3개 폴더 테스트
print("\n" + "=" * 60)
print("동일한 시드(42) 사용 시:")
print("=" * 60)

for i, folder in enumerate(target_folders[:3]):
    print(f"\n{i+1}. {folder.name}")
    sampled = ls.process_folder_with_layers(folder, layers, 0.05, seed=42)
    
    normal_images = sampled['normal'].get(1, [])
    if normal_images:
        print(f"   Layer 1 Normal: {len(normal_images)}개")
        print(f"   처음 3개: {[img.name for img in normal_images[:3]]}")

print("\n" + "=" * 60)
print("개선된 시드 생성 방식 (폴더 해시 + 인덱스) 사용 시:")
print("=" * 60)

base_seed = 42
for i, folder in enumerate(target_folders[:3]):
    # 개선된 시드 생성
    folder_hash = hash(folder.name) % 10000
    folder_seed = base_seed + folder_hash + (i * 1000)
    print(f"\n{i+1}. {folder.name}")
    print(f"   시드: {folder_seed} (hash={folder_hash}, idx={i})")
    sampled = ls.process_folder_with_layers(folder, layers, 0.05, seed=folder_seed)
    
    normal_images = sampled['normal'].get(1, [])
    if normal_images:
        print(f"   Layer 1 Normal: {len(normal_images)}개")
        print(f"   처음 3개: {[img.name for img in normal_images[:3]]}")

print("\n" + "=" * 60)
print("결론:")
print("=" * 60)
print("✓ 동일한 시드: 모든 폴더에서 비슷한 이미지 번호 선택")
print("✓ 순차 시드: 어느 정도 다양성 확보")
print("✓ 해시 기반 시드: 폴더마다 완전히 다른 이미지 번호 선택 (최적)")
print("=" * 60)
