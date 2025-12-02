#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer Analyzer - 딥러닝 모델 기반 자동 층 분석 도구

NRT 모델을 사용하여 코일 감기 이미지의 Y좌표를 추적하고,
Savitzky-Golay 필터로 추세선을 분석하여 층 전환점을 자동 감지합니다.

출력: layers.json (layer_sampler_gui.py에서 사용)
"""

import os
import sys
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks, savgol_filter

# NRT import
try:
    import nrt
    NRT_AVAILABLE = True
except ImportError:
    NRT_AVAILABLE = False
    print("Warning: NRT 모듈을 찾을 수 없습니다. 모델 추론이 불가능합니다.")

import cv2


def imread_korean(filepath):
    """한글 경로 지원 이미지 읽기"""
    try:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"이미지 읽기 오류: {filepath} - {e}")
        return None


def preprocess_image(img, target_size=512):
    """이미지 전처리 (모델 입력 크기로 리사이즈)"""
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    # 정사각형으로 패딩
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas, scale, x_offset, y_offset


def extract_frame_number(filename):
    """파일명에서 프레임 번호 추출 (예: A_000123.jpg -> 123)"""
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    # 숫자만 있는 경우
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def analyze_folder(folder_path, model_path, progress_callback=None):
    """
    단일 폴더 분석 - Y좌표 추적 및 층 분석
    
    Args:
        folder_path: 이미지 폴더 경로
        model_path: NRT 모델 경로
        progress_callback: 진행률 콜백 함수 (current, total, message)
    
    Returns:
        dict: 층 정보 리스트 또는 None (실패시)
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
        return None
    
    # 이미지 파일 목록
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted([
        f for f in folder_path.iterdir()
        if f.suffix.lower() in image_extensions
    ], key=lambda x: extract_frame_number(x.name))
    
    if not images:
        print(f"이미지를 찾을 수 없습니다: {folder_path}")
        return None
    
    print(f"폴더: {folder_path.name}")
    print(f"이미지 수: {len(images)}개")
    
    # NRT 모델 로드 (GPU 사용)
    if not NRT_AVAILABLE:
        print("NRT를 사용할 수 없습니다.")
        return None
    
    try:
        # GPU로 Predictor 생성
        # 파라미터: model_path, modelio_flag, device_idx, batch_size, fp16, threshold, DevType
        predictor = nrt.Predictor(
            str(model_path),
            nrt.Model.MODELIO_DEFAULT,  # modelio_flag
            0,                          # device_idx (GPU 0)
            1,                          # batch_size
            False,                      # fp16_flag
            True,                       # threshold_flag
            nrt.DEVICE_CUDA_GPU         # GPU 사용!
        )
        dev_type = predictor.get_device_type()
        if dev_type == nrt.DEVICE_CUDA_GPU:
            print(f"  GPU 모드로 실행 (CUDA)")
        else:
            print(f"  CPU 모드로 실행 (device_type={dev_type})")
    except Exception as e:
        print(f"GPU 모델 로드 실패, CPU로 재시도: {e}")
        try:
            predictor = nrt.Predictor(str(model_path))
        except Exception as e2:
            print(f"모델 로드 실패: {e2}")
            return None
    
    # 추론 실행
    tracking_data = []
    total = len(images)
    
    for idx, img_path in enumerate(images):
        if progress_callback:
            progress_callback(idx + 1, total, f"추론 중: {img_path.name}")
        
        # 이미지 읽기 및 전처리
        img = imread_korean(str(img_path))
        if img is None:
            continue
        
        result_data = preprocess_image(img, 512)
        if result_data is None:
            continue
        
        processed_img, scale, x_offset, y_offset = result_data
        
        # 추론
        try:
            input_data = nrt.Input()
            image_buff = nrt.NDBuffer.from_numpy(processed_img)
            input_data.extend(image_buff)
            result = predictor.predict(input_data)
            input_data.clear()
            
            # 결과 파싱
            if result.bboxes.get_count() > 0:
                bbox = result.bboxes.get(0)  # 첫 번째 검출
                
                # 원본 좌표로 변환
                orig_x = (bbox.rect.x - x_offset) / scale
                orig_y = (bbox.rect.y - y_offset) / scale
                orig_w = bbox.rect.width / scale
                orig_h = bbox.rect.height / scale
                
                center_x = orig_x + orig_w / 2
                center_y = orig_y + orig_h / 2
                
                frame_num = extract_frame_number(img_path.name)
                
                tracking_data.append({
                    'frame': frame_num,
                    'filename': img_path.name,
                    'center_x': center_x,
                    'center_y': center_y
                })
        except Exception as e:
            print(f"추론 오류 ({img_path.name}): {e}")
            continue
        
        # 진행률 출력 (10% 단위)
        if (idx + 1) % max(1, total // 10) == 0:
            print(f"  진행: {idx + 1}/{total} ({100 * (idx + 1) // total}%)")
    
    print(f"  감지된 프레임: {len(tracking_data)}개 / {total}개")
    
    if len(tracking_data) < 10:
        print("  감지된 프레임이 너무 적습니다.")
        return None
    
    # 층 분석
    layers = analyze_layers(tracking_data, folder_path.name)
    
    return layers


def analyze_layers(tracking_data, folder_name):
    """
    Y좌표 데이터를 분석하여 층 구간 계산
    
    Args:
        tracking_data: 추적 데이터 리스트
        folder_name: 폴더명 (로그용)
    
    Returns:
        list: 층 정보 리스트
    """
    if len(tracking_data) < 50:
        print(f"  데이터가 부족합니다: {len(tracking_data)}개")
        return None
    
    # 데이터 정렬 (프레임 번호 순)
    tracking_data = sorted(tracking_data, key=lambda x: x['frame'])
    
    frames = np.array([d['frame'] for d in tracking_data])
    y_values = np.array([d['center_y'] for d in tracking_data])
    
    # Savitzky-Golay 필터로 추세선 계산
    # 윈도우 크기는 데이터 크기에 따라 조정
    window_length = min(101, len(y_values) // 3)
    if window_length % 2 == 0:
        window_length += 1  # 홀수로 맞춤
    window_length = max(5, window_length)
    
    poly_order = min(3, window_length - 1)
    
    y_smooth = savgol_filter(y_values, window_length, poly_order)
    
    # 피크(상단 전환점) 찾기
    # distance: 최소 피크 간격 (프레임 수 기준)
    # prominence: 피크의 최소 높이 차이
    min_distance = max(50, len(y_values) // 30)  # 최소 50 또는 데이터의 1/30
    
    # Y 범위의 10%를 prominence로 사용
    y_range = np.max(y_smooth) - np.min(y_smooth)
    prominence = y_range * 0.15
    
    peaks_max, _ = find_peaks(y_smooth, distance=min_distance, prominence=prominence)
    peaks_min, _ = find_peaks(-y_smooth, distance=min_distance, prominence=prominence)
    
    print(f"  분석 파라미터: window={window_length}, prominence={prominence:.1f}")
    print(f"  상단 전환점: {len(peaks_max)}개, 하단 전환점: {len(peaks_min)}개")
    
    # 모든 전환점 합치고 정렬
    all_peaks = sorted(list(peaks_max) + list(peaks_min))
    num_layers = len(all_peaks) + 1
    
    print(f"  추정 층 수: {num_layers}층")
    
    # 층별 구간 계산
    layers = []
    prev_idx = 0
    
    for i, peak_idx in enumerate(all_peaks):
        start_frame = int(frames[prev_idx])
        end_frame = int(frames[peak_idx])
        count = peak_idx - prev_idx + 1
        
        layers.append({
            'layer': i + 1,
            'start': start_frame,
            'end': end_frame,
            'count': count
        })
        prev_idx = peak_idx
    
    # 마지막 층
    layers.append({
        'layer': len(all_peaks) + 1,
        'start': int(frames[prev_idx]),
        'end': int(frames[-1]),
        'count': len(frames) - prev_idx
    })
    
    return layers


def find_target_folders(root_path, keywords=None):
    """
    대상 폴더 찾기
    
    Args:
        root_path: 루트 경로
        keywords: 검색 키워드 리스트 (None이면 모든 하위 폴더)
    
    Returns:
        list: 폴더 경로 리스트
    """
    root_path = Path(root_path)
    
    if not root_path.exists():
        return []
    
    folders = []
    
    for item in root_path.iterdir():
        if item.is_dir():
            # 키워드 필터링
            if keywords:
                if all(kw.lower() in item.name.lower() for kw in keywords):
                    folders.append(item)
            else:
                # 이미지가 있는 폴더만
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                has_images = any(
                    f.suffix.lower() in image_extensions
                    for f in item.iterdir() if f.is_file()
                )
                if has_images:
                    folders.append(item)
    
    return sorted(folders, key=lambda x: x.name)


def analyze_all_folders(root_path, model_path, output_path, keywords=None, progress_callback=None):
    """
    전체 폴더 분석 및 layers.json 생성
    
    Args:
        root_path: 루트 경로
        model_path: NRT 모델 경로
        output_path: 출력 JSON 파일 경로
        keywords: 검색 키워드 리스트
        progress_callback: 진행률 콜백 (folder_idx, total_folders, folder_name)
    
    Returns:
        dict: 전체 분석 결과
    """
    root_path = Path(root_path)
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    # 모델 확인
    if not model_path.exists():
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    # 대상 폴더 찾기
    folders = find_target_folders(root_path, keywords)
    
    if not folders:
        print("분석할 폴더를 찾을 수 없습니다.")
        return None
    
    print(f"\n{'='*60}")
    print(f"Layer Analyzer - 전체 폴더 분석")
    print(f"{'='*60}")
    print(f"루트 경로: {root_path}")
    print(f"모델 경로: {model_path}")
    print(f"대상 폴더: {len(folders)}개")
    print(f"{'='*60}\n")
    
    # 전체 결과 저장
    all_results = {}
    
    for folder_idx, folder in enumerate(folders):
        print(f"\n[{folder_idx + 1}/{len(folders)}] {folder.name}")
        print("-" * 50)
        
        if progress_callback:
            progress_callback(folder_idx + 1, len(folders), folder.name)
        
        # 폴더 분석
        layers = analyze_folder(folder, model_path)
        
        if layers:
            all_results[folder.name] = layers
            print(f"  ✓ 완료: {len(layers)}층 감지")
        else:
            print(f"  ✗ 실패: 층 분석 불가")
    
    # 결과 저장
    if all_results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"분석 완료!")
        print(f"{'='*60}")
        print(f"성공: {len(all_results)}/{len(folders)} 폴더")
        print(f"출력 파일: {output_path}")
        print(f"{'='*60}")
    
    return all_results


# ============================================================
# GUI 버전
# ============================================================
def run_gui():
    """PyQt5 GUI 실행"""
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
        QProgressBar, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
        QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
        QSplitter, QComboBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QProcess
    from PyQt5.QtGui import QColor
    import subprocess
    
    class AnalyzerThread(QThread):
        """백그라운드 분석 스레드"""
        progress = pyqtSignal(int, int, str)  # current, total, message
        folder_progress = pyqtSignal(int, int, str)  # folder_idx, total, folder_name
        log = pyqtSignal(str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)
        
        def __init__(self, root_path, model_path, output_path, keywords=None):
            super().__init__()
            self.root_path = root_path
            self.model_path = model_path
            self.output_path = output_path
            self.keywords = keywords
            self._is_running = True
        
        def stop(self):
            self._is_running = False
        
        def run(self):
            try:
                result = analyze_all_folders(
                    self.root_path,
                    self.model_path,
                    self.output_path,
                    self.keywords,
                    progress_callback=lambda i, t, m: self.folder_progress.emit(i, t, m)
                )
                if result:
                    self.finished.emit(result)
                else:
                    self.error.emit("분석 결과가 없습니다.")
            except Exception as e:
                self.error.emit(str(e))
    
    class LayerAnalyzerGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.analyzer_thread = None
            self.analysis_result = {}  # 분석 결과 저장
            self.init_ui()
        
        def init_ui(self):
            self.setWindowTitle("Layer Analyzer - 딥러닝 기반 층 자동 분석 & 샘플링 연계")
            self.setGeometry(100, 100, 1200, 800)
            
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            main_layout = QVBoxLayout()
            
            # 탭 위젯
            self.tab_widget = QTabWidget()
            
            # ========== 탭 1: 추론 실행 ==========
            inference_tab = QWidget()
            inference_layout = QVBoxLayout()
            
            # 입력 설정
            input_group = QGroupBox("입력 설정")
            input_layout = QGridLayout()
            
            # 루트 폴더
            input_layout.addWidget(QLabel("이미지 루트 폴더:"), 0, 0)
            self.root_edit = QLineEdit()
            self.root_edit.setText("K:/LLM Image_Storage")
            input_layout.addWidget(self.root_edit, 0, 1)
            root_btn = QPushButton("찾아보기")
            root_btn.clicked.connect(self.browse_root)
            input_layout.addWidget(root_btn, 0, 2)
            
            # 모델 파일
            input_layout.addWidget(QLabel("NRT 모델 파일:"), 1, 0)
            self.model_edit = QLineEdit()
            self.model_edit.setText("D:/LLM_Dataset/models/Trace_Coil.net")
            input_layout.addWidget(self.model_edit, 1, 1)
            model_btn = QPushButton("찾아보기")
            model_btn.clicked.connect(self.browse_model)
            input_layout.addWidget(model_btn, 1, 2)
            
            # 출력 파일
            input_layout.addWidget(QLabel("출력 JSON 파일:"), 2, 0)
            self.output_edit = QLineEdit()
            self.output_edit.setText("D:/LLM_Dataset/layers.json")
            input_layout.addWidget(self.output_edit, 2, 1)
            output_btn = QPushButton("찾아보기")
            output_btn.clicked.connect(self.browse_output)
            input_layout.addWidget(output_btn, 2, 2)
            
            # 키워드 필터
            input_layout.addWidget(QLabel("폴더 키워드 (콤마 구분):"), 3, 0)
            self.keywords_edit = QLineEdit()
            self.keywords_edit.setPlaceholderText("예: A line (비워두면 모든 폴더)")
            input_layout.addWidget(self.keywords_edit, 3, 1, 1, 2)
            
            input_group.setLayout(input_layout)
            inference_layout.addWidget(input_group)
            
            # 분석 파라미터
            param_group = QGroupBox("분석 파라미터")
            param_layout = QHBoxLayout()
            
            param_layout.addWidget(QLabel("Savitzky-Golay Window:"))
            self.window_spin = QSpinBox()
            self.window_spin.setRange(5, 201)
            self.window_spin.setValue(101)
            self.window_spin.setSingleStep(2)
            param_layout.addWidget(self.window_spin)
            
            param_layout.addWidget(QLabel("Prominence (%):"))
            self.prominence_spin = QDoubleSpinBox()
            self.prominence_spin.setRange(5, 50)
            self.prominence_spin.setValue(15)
            self.prominence_spin.setSingleStep(1)
            param_layout.addWidget(self.prominence_spin)
            
            param_layout.addStretch()
            param_group.setLayout(param_layout)
            inference_layout.addWidget(param_group)
            
            # 실행 버튼
            btn_layout = QHBoxLayout()
            self.start_btn = QPushButton("🚀 분석 시작")
            self.start_btn.clicked.connect(self.start_analysis)
            self.start_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; padding: 10px; font-size: 14px; }")
            btn_layout.addWidget(self.start_btn)
            
            self.stop_btn = QPushButton("⏹ 중지")
            self.stop_btn.clicked.connect(self.stop_analysis)
            self.stop_btn.setEnabled(False)
            btn_layout.addWidget(self.stop_btn)
            
            inference_layout.addLayout(btn_layout)
            
            # 진행률
            self.progress_bar = QProgressBar()
            self.progress_bar.setFormat("%v / %m 폴더 (%p%)")
            inference_layout.addWidget(self.progress_bar)
            
            self.status_label = QLabel("대기 중...")
            inference_layout.addWidget(self.status_label)
            
            # 로그
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            inference_layout.addWidget(self.log_text)
            
            inference_tab.setLayout(inference_layout)
            self.tab_widget.addTab(inference_tab, "1. 추론 실행")
            
            # ========== 탭 2: 분석 결과 ==========
            result_tab = QWidget()
            result_layout = QVBoxLayout()
            
            # 결과 파일 로드
            load_group = QGroupBox("분석 결과 파일")
            load_layout = QHBoxLayout()
            
            self.result_file_edit = QLineEdit()
            self.result_file_edit.setText("D:/LLM_Dataset/layers.json")
            load_layout.addWidget(self.result_file_edit)
            
            load_btn = QPushButton("불러오기")
            load_btn.clicked.connect(self.load_result_file)
            load_layout.addWidget(load_btn)
            
            browse_result_btn = QPushButton("찾아보기")
            browse_result_btn.clicked.connect(self.browse_result_file)
            load_layout.addWidget(browse_result_btn)
            
            load_group.setLayout(load_layout)
            result_layout.addWidget(load_group)
            
            # 폴더별 요약 테이블
            summary_group = QGroupBox("폴더별 분석 요약")
            summary_layout = QVBoxLayout()
            
            self.summary_table = QTableWidget()
            self.summary_table.setColumnCount(4)
            self.summary_table.setHorizontalHeaderLabels(["폴더명", "층 수", "총 프레임", "상태"])
            self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.summary_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.summary_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.summary_table.itemSelectionChanged.connect(self.on_folder_selected)
            summary_layout.addWidget(self.summary_table)
            
            summary_group.setLayout(summary_layout)
            result_layout.addWidget(summary_group)
            
            # 선택된 폴더의 층 상세 정보
            detail_group = QGroupBox("선택된 폴더 - 층별 상세 정보")
            detail_layout = QVBoxLayout()
            
            self.detail_table = QTableWidget()
            self.detail_table.setColumnCount(4)
            self.detail_table.setHorizontalHeaderLabels(["층", "시작 프레임", "종료 프레임", "이미지 수"])
            self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            detail_layout.addWidget(self.detail_table)
            
            detail_group.setLayout(detail_layout)
            result_layout.addWidget(detail_group)
            
            # 결과 수정/저장 버튼
            edit_btn_layout = QHBoxLayout()
            
            self.save_result_btn = QPushButton("💾 결과 저장")
            self.save_result_btn.clicked.connect(self.save_result)
            edit_btn_layout.addWidget(self.save_result_btn)
            
            self.export_csv_btn = QPushButton("📊 CSV 내보내기")
            self.export_csv_btn.clicked.connect(self.export_to_csv)
            edit_btn_layout.addWidget(self.export_csv_btn)
            
            edit_btn_layout.addStretch()
            result_layout.addLayout(edit_btn_layout)
            
            result_tab.setLayout(result_layout)
            self.tab_widget.addTab(result_tab, "2. 분석 결과")
            
            # ========== 탭 3: 샘플링 연계 ==========
            sampling_tab = QWidget()
            sampling_layout = QVBoxLayout()
            
            # Layer Sampler 설정
            sampler_group = QGroupBox("Layer Sampler GUI 연계")
            sampler_layout = QGridLayout()
            
            sampler_layout.addWidget(QLabel("Layer Sampler 경로:"), 0, 0)
            self.sampler_path_edit = QLineEdit()
            self.sampler_path_edit.setText("D:/LLM_Dataset/layer_sampler_gui.py")
            sampler_layout.addWidget(self.sampler_path_edit, 0, 1)
            
            browse_sampler_btn = QPushButton("찾아보기")
            browse_sampler_btn.clicked.connect(self.browse_sampler)
            sampler_layout.addWidget(browse_sampler_btn, 0, 2)
            
            sampler_layout.addWidget(QLabel("사용할 layers.json:"), 1, 0)
            self.layers_json_edit = QLineEdit()
            self.layers_json_edit.setText("D:/LLM_Dataset/layers.json")
            sampler_layout.addWidget(self.layers_json_edit, 1, 1)
            
            copy_path_btn = QPushButton("결과 파일 복사")
            copy_path_btn.clicked.connect(lambda: self.layers_json_edit.setText(self.result_file_edit.text()))
            sampler_layout.addWidget(copy_path_btn, 1, 2)
            
            sampler_group.setLayout(sampler_layout)
            sampling_layout.addWidget(sampler_group)
            
            # 실행 버튼
            run_sampler_layout = QHBoxLayout()
            
            self.run_sampler_btn = QPushButton("🎯 Layer Sampler GUI 실행")
            self.run_sampler_btn.clicked.connect(self.run_layer_sampler)
            self.run_sampler_btn.setStyleSheet("QPushButton { background-color: #107c10; color: white; padding: 15px; font-size: 16px; }")
            run_sampler_layout.addWidget(self.run_sampler_btn)
            
            sampling_layout.addLayout(run_sampler_layout)
            
            # 워크플로우 안내
            workflow_group = QGroupBox("워크플로우 안내")
            workflow_layout = QVBoxLayout()
            
            workflow_text = QTextEdit()
            workflow_text.setReadOnly(True)
            workflow_text.setMaximumHeight(200)
            workflow_text.setHtml("""
            <h3>📋 사용 순서</h3>
            <ol>
                <li><b>추론 실행</b>: 이미지 폴더를 선택하고 딥러닝 모델로 층 분석 실행</li>
                <li><b>분석 결과 확인</b>: 각 폴더별 층 정보 확인 및 필요시 수정</li>
                <li><b>샘플링 연계</b>: Layer Sampler GUI 실행하여 층별 이미지 샘플링</li>
            </ol>
            <h3>📁 출력 파일</h3>
            <ul>
                <li><b>layers.json</b>: 폴더별 층 구간 정보 (Layer Sampler에서 사용)</li>
            </ul>
            <h3>⚙️ 파라미터 설명</h3>
            <ul>
                <li><b>Savitzky-Golay Window</b>: 추세선 스무딩 윈도우 크기 (클수록 부드러움)</li>
                <li><b>Prominence</b>: 전환점 감지 민감도 (작을수록 더 많은 전환점 감지)</li>
            </ul>
            """)
            workflow_layout.addWidget(workflow_text)
            
            workflow_group.setLayout(workflow_layout)
            sampling_layout.addWidget(workflow_group)
            
            sampling_layout.addStretch()
            sampling_tab.setLayout(sampling_layout)
            self.tab_widget.addTab(sampling_tab, "3. 샘플링 연계")
            
            main_layout.addWidget(self.tab_widget)
            main_widget.setLayout(main_layout)
            
            # 스타일
            self.setStyleSheet("""
                QMainWindow { background-color: #1e1e1e; }
                QWidget { background-color: #1e1e1e; color: #d4d4d4; }
                QGroupBox { border: 2px solid #3c3c3c; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
                QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #3c3c3c; border: 1px solid #555; padding: 5px; }
                QTextEdit { background-color: #252526; border: 1px solid #3c3c3c; }
                QPushButton { background-color: #3c3c3c; border: 1px solid #555; padding: 8px; }
                QPushButton:hover { background-color: #4c4c4c; }
                QPushButton:disabled { background-color: #2d2d2d; color: #666; }
                QTabWidget::pane { border: 1px solid #3c3c3c; }
                QTabBar::tab { background-color: #2d2d2d; padding: 10px 20px; margin-right: 2px; }
                QTabBar::tab:selected { background-color: #3c3c3c; }
                QTableWidget { background-color: #252526; gridline-color: #3c3c3c; }
                QHeaderView::section { background-color: #3c3c3c; padding: 5px; border: 1px solid #555; }
            """)
        
        def browse_root(self):
            folder = QFileDialog.getExistingDirectory(self, "이미지 루트 폴더 선택")
            if folder:
                self.root_edit.setText(folder)
        
        def browse_model(self):
            file, _ = QFileDialog.getOpenFileName(self, "NRT 모델 선택", "", "NRT Model (*.net)")
            if file:
                self.model_edit.setText(file)
        
        def browse_output(self):
            file, _ = QFileDialog.getSaveFileName(self, "출력 JSON 파일", "", "JSON (*.json)")
            if file:
                self.output_edit.setText(file)
        
        def browse_result_file(self):
            file, _ = QFileDialog.getOpenFileName(self, "분석 결과 파일 선택", "", "JSON (*.json)")
            if file:
                self.result_file_edit.setText(file)
                self.load_result_file()
        
        def browse_sampler(self):
            file, _ = QFileDialog.getOpenFileName(self, "Layer Sampler 선택", "", "Python (*.py)")
            if file:
                self.sampler_path_edit.setText(file)
        
        def log(self, message):
            self.log_text.append(message)
        
        def load_result_file(self):
            """분석 결과 파일 로드"""
            file_path = self.result_file_edit.text()
            if not file_path or not Path(file_path).exists():
                QMessageBox.warning(self, "경고", "파일을 찾을 수 없습니다.")
                return
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.analysis_result = json.load(f)
                
                self.update_summary_table()
                QMessageBox.information(self, "완료", f"{len(self.analysis_result)}개 폴더 정보를 로드했습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"파일 로드 실패: {e}")
        
        def update_summary_table(self):
            """폴더별 요약 테이블 업데이트"""
            self.summary_table.setRowCount(len(self.analysis_result))
            
            for row, (folder_name, layers) in enumerate(self.analysis_result.items()):
                # 폴더명
                self.summary_table.setItem(row, 0, QTableWidgetItem(folder_name))
                
                # 층 수
                num_layers = len(layers)
                self.summary_table.setItem(row, 1, QTableWidgetItem(str(num_layers)))
                
                # 총 프레임
                total_frames = sum(layer.get('count', 0) for layer in layers)
                self.summary_table.setItem(row, 2, QTableWidgetItem(str(total_frames)))
                
                # 상태 (층 수에 따라 색상)
                status_item = QTableWidgetItem("✓ 정상" if 10 <= num_layers <= 25 else "⚠ 확인필요")
                if num_layers < 10 or num_layers > 25:
                    status_item.setBackground(QColor(100, 50, 50))
                else:
                    status_item.setBackground(QColor(50, 100, 50))
                self.summary_table.setItem(row, 3, status_item)
        
        def on_folder_selected(self):
            """폴더 선택 시 상세 정보 표시"""
            selected_rows = self.summary_table.selectedItems()
            if not selected_rows:
                return
            
            row = selected_rows[0].row()
            folder_name = self.summary_table.item(row, 0).text()
            
            if folder_name not in self.analysis_result:
                return
            
            layers = self.analysis_result[folder_name]
            self.detail_table.setRowCount(len(layers))
            
            for i, layer in enumerate(layers):
                self.detail_table.setItem(i, 0, QTableWidgetItem(str(layer.get('layer', i+1))))
                self.detail_table.setItem(i, 1, QTableWidgetItem(str(layer.get('start', ''))))
                self.detail_table.setItem(i, 2, QTableWidgetItem(str(layer.get('end', ''))))
                self.detail_table.setItem(i, 3, QTableWidgetItem(str(layer.get('count', ''))))
        
        def save_result(self):
            """분석 결과 저장"""
            if not self.analysis_result:
                QMessageBox.warning(self, "경고", "저장할 결과가 없습니다.")
                return
            
            file_path = self.result_file_edit.text()
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_result, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "완료", f"저장 완료: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 실패: {e}")
        
        def export_to_csv(self):
            """CSV 내보내기"""
            if not self.analysis_result:
                QMessageBox.warning(self, "경고", "내보낼 결과가 없습니다.")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", "", "CSV (*.csv)")
            if not file_path:
                return
            
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(['폴더명', '층', '시작 프레임', '종료 프레임', '이미지 수'])
                    
                    for folder_name, layers in self.analysis_result.items():
                        for layer in layers:
                            writer.writerow([
                                folder_name,
                                layer.get('layer', ''),
                                layer.get('start', ''),
                                layer.get('end', ''),
                                layer.get('count', '')
                            ])
                
                QMessageBox.information(self, "완료", f"CSV 내보내기 완료: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"내보내기 실패: {e}")
        
        def run_layer_sampler(self):
            """Layer Sampler GUI 실행"""
            sampler_path = self.sampler_path_edit.text()
            layers_json = self.layers_json_edit.text()
            
            if not Path(sampler_path).exists():
                QMessageBox.warning(self, "경고", "Layer Sampler 파일을 찾을 수 없습니다.")
                return
            
            if not Path(layers_json).exists():
                QMessageBox.warning(self, "경고", "layers.json 파일을 찾을 수 없습니다.\n먼저 추론을 실행하거나 결과 파일을 선택해주세요.")
                return
            
            try:
                # Layer Sampler GUI 실행
                python_exe = sys.executable
                subprocess.Popen([python_exe, sampler_path], cwd=str(Path(sampler_path).parent))
                self.log(f"Layer Sampler GUI 실행: {sampler_path}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"실행 실패: {e}")
        
        def start_analysis(self):
            root_path = self.root_edit.text()
            model_path = self.model_edit.text()
            output_path = self.output_edit.text()
            
            if not root_path or not model_path or not output_path:
                QMessageBox.warning(self, "경고", "모든 경로를 입력해주세요.")
                return
            
            if not Path(root_path).exists():
                QMessageBox.warning(self, "경고", "루트 폴더를 찾을 수 없습니다.")
                return
            
            if not Path(model_path).exists():
                QMessageBox.warning(self, "경고", "모델 파일을 찾을 수 없습니다.")
                return
            
            # 키워드 파싱
            keywords_text = self.keywords_edit.text().strip()
            keywords = [k.strip() for k in keywords_text.split(',')] if keywords_text else None
            
            self.log_text.clear()
            self.log(f"분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log(f"루트 폴더: {root_path}")
            self.log(f"모델: {model_path}")
            self.log(f"키워드: {keywords}")
            
            # 스레드 시작
            self.analyzer_thread = AnalyzerThread(root_path, model_path, output_path, keywords)
            self.analyzer_thread.folder_progress.connect(self.on_folder_progress)
            self.analyzer_thread.finished.connect(self.on_finished)
            self.analyzer_thread.error.connect(self.on_error)
            self.analyzer_thread.start()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        
        def stop_analysis(self):
            if self.analyzer_thread:
                self.analyzer_thread.stop()
                self.log("분석 중지 요청...")
        
        def on_folder_progress(self, current, total, folder_name):
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.status_label.setText(f"분석 중: {folder_name}")
            self.log(f"[{current}/{total}] {folder_name}")
        
        def on_finished(self, result):
            self.analysis_result = result  # 결과 저장
            self.log(f"\n분석 완료: {len(result)}개 폴더")
            self.log(f"저장: {self.output_edit.text()}")
            self.status_label.setText(f"완료: {len(result)}개 폴더 분석됨")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            # 결과 파일 경로 동기화
            self.result_file_edit.setText(self.output_edit.text())
            self.layers_json_edit.setText(self.output_edit.text())
            
            # 요약 테이블 업데이트
            self.update_summary_table()
            
            # 결과 탭으로 이동
            self.tab_widget.setCurrentIndex(1)
            
            QMessageBox.information(self, "완료", f"분석 완료!\n{len(result)}개 폴더의 층 정보가 저장되었습니다.\n\n결과 탭에서 확인하세요.")
        
        def on_error(self, error_msg):
            self.log(f"\n오류: {error_msg}")
            self.status_label.setText(f"오류 발생")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            QMessageBox.critical(self, "오류", error_msg)
    
    app = QApplication(sys.argv)
    window = LayerAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())


# ============================================================
# CLI 버전
# ============================================================
def run_cli():
    """명령줄 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Layer Analyzer - 딥러닝 기반 층 자동 분석')
    parser.add_argument('--root', '-r', required=True, help='이미지 루트 폴더')
    parser.add_argument('--model', '-m', required=True, help='NRT 모델 파일 경로')
    parser.add_argument('--output', '-o', default='layers.json', help='출력 JSON 파일')
    parser.add_argument('--keywords', '-k', nargs='*', help='폴더 검색 키워드')
    parser.add_argument('--single', '-s', help='단일 폴더만 분석')
    
    args = parser.parse_args()
    
    if args.single:
        # 단일 폴더 분석
        layers = analyze_folder(args.single, args.model)
        if layers:
            folder_name = Path(args.single).name
            result = {folder_name: layers}
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n저장 완료: {args.output}")
    else:
        # 전체 폴더 분석
        analyze_all_folders(args.root, args.model, args.output, args.keywords)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_cli()
    else:
        run_gui()
