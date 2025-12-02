#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer Boundary Detector - 권취 라인 층 경계 자동 탐지

이미지 높이 변화를 분석하여 층 경계를 자동으로 찾습니다.
각 폴더마다 고유한 layers.json을 생성합니다.
"""
import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
# 한글 폰트 설정 (안전하게)
try:
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        matplotlib.rcParams['font.family'] = 'NanumGothic'
    except:
        pass  # 기본 폰트 사용
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 표시
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QSpinBox,
    QGroupBox, QGridLayout, QMessageBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QDialog, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageAnalyzerThread(QThread):
    """이미지 분석 스레드"""
    progress = pyqtSignal(str, int)  # 메시지, 진행률
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)
    
    def run(self):
        try:
            result = self.analyze_images()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def analyze_images(self):
        """이미지 분석 - 밝기 프로파일 및 Z-Projection"""
        self.progress.emit("이미지 파일 검색 중...", 0)
        
        # 이미지 파일 찾기 (정상/불량 폴더 모두 포함)
        image_files = []
        for ext in IMG_EXTS:
            image_files.extend(self.folder_path.rglob(f"*{ext}"))
        
        # 자연스러운 정렬 (숫자 순서)
        import re
        def natural_key(s):
            return [int(t) if t.isdigit() else t.lower() 
                    for t in re.split(r'(\d+)', str(s))]
        
        image_files = sorted(image_files, key=natural_key)
        
        if not image_files:
            raise ValueError("이미지 파일을 찾을 수 없습니다.")
        
        total = len(image_files)
        self.progress.emit(f"총 {total}개 이미지 분석 시작...", 5)
        
        # 각 이미지 분석: 높이, 밝기, 중앙선 추출
        heights = []
        brightness_profile = []
        center_lines = []  # Z-Projection용 중앙 수직선
        failed_files = []  # 실패한 파일 추적
        
        for idx, img_path in enumerate(image_files):
            try:
                # 파일 존재 확인
                if not img_path.exists():
                    self.progress.emit(f"경고: {img_path.name} 파일이 존재하지 않음", -1)
                    continue
                
                with Image.open(img_path) as img:
                    # 그레이스케일 변환
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    img_array = np.array(img)
                    width, height = img.size
                    
                    # 중앙 수직선 추출 (측면 뷰용)
                    center_x = width // 2
                    center_line = img_array[:, center_x]
                    center_lines.append(center_line)
                    
                    # 평균 밝기 계산
                    avg_brightness = np.mean(img_array)
                    
                    # 에지 강도 (수평 gradient)
                    gradient = np.abs(np.diff(img_array.astype(float), axis=0))
                    edge_strength = np.mean(gradient)
                    # Seam detection 제거 - Z-profile에만 집중
                    
                    heights.append({
                        'index': idx + 1,
                        'path': img_path,
                        'width': width,
                        'height': height
                    })
                    
                    brightness_profile.append({
                        'index': idx + 1,
                        'brightness': avg_brightness,
                        'edge_strength': edge_strength
                    })
                
                if (idx + 1) % 100 == 0:
                    progress_pct = int((idx + 1) / total * 80) + 5
                    self.progress.emit(f"분석 중: {idx + 1}/{total}", progress_pct)
            
            except Exception as e:
                error_type = type(e).__name__
                failed_files.append({'name': img_path.name, 'error': f"{error_type}: {str(e)[:50]}"})
                self.progress.emit(f"경고: {img_path.name} 읽기 실패 ({error_type}: {str(e)[:50]})", -1)
                continue
        
        self.progress.emit("Z-Projection 생성 중...", 85)
        
        # Z-Projection: 측면 이미지 생성
        if center_lines:
            max_height = max(len(line) for line in center_lines)
            # 모든 라인을 같은 높이로 맞추기 (패딩)
            padded_lines = []
            for line in center_lines:
                if len(line) < max_height:
                    padded = np.pad(line, (0, max_height - len(line)), mode='constant', constant_values=0)
                    padded_lines.append(padded)
                else:
                    padded_lines.append(line)
            
            # 측면 이미지 생성 (가로: 이미지 개수, 세로: 이미지 높이)
            z_projection = np.column_stack(padded_lines)
        else:
            z_projection = None
        
        self.progress.emit("대각선 꼭지점 탐지 중...", 90)
        
        # Z-Projection에서 각 열(이미지 인덱스)의 최대 밝기 위치 추적
        # 대각선이 올라가다가 꼭지점(피크)에서 다시 내려감
        peak_positions = []
        z_profile = None
        detection_params = {}  # 탐지 파라미터 저장
        
        if z_projection is not None and z_projection.shape[1] > 10:
            # 각 열에서 가장 밝은 픽셀의 Y 위치 찾기
            max_brightness_y = []
            for col_idx in range(z_projection.shape[1]):
                column = z_projection[:, col_idx]
                if column.max() > 50:  # 충분히 밝은 픽셀이 있는 경우만
                    max_y = np.argmax(column)
                    max_brightness_y.append(max_y)
                else:
                    max_brightness_y.append(None)
            
            if max_brightness_y.count(None) < len(max_brightness_y):
                y_array = np.array([
                    float(val) if val is not None else np.nan for val in max_brightness_y
                ])
                valid_mask = ~np.isnan(y_array)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) >= 5:
                    indices = np.arange(len(y_array))
                    filled_values = np.copy(y_array)
                    filled_values[~valid_mask] = np.interp(
                        indices[~valid_mask], indices[valid_mask], y_array[valid_mask]
                    )
                    
                    window = 11
                    smoothed = np.convolve(filled_values, np.ones(window)/window, mode='same')
                    dy = np.diff(smoothed)
                    
                    z_profile = {
                        'raw': filled_values.tolist(),
                        'smoothed': smoothed.tolist(),
                        'slope': dy.tolist()
                    }
            
            # Hough Line 기반 빗변 검출
            detected_lines = []
            if z_projection is not None:
                self.progress.emit("Hough Line 기반 빗변 검출 중...", 80)
                
                # 8-bit 변환
                z_proj_norm = cv2.normalize(z_projection, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Canny 에지 검출 (더 엄격하게)
                edges = cv2.Canny(z_proj_norm, 50, 150)
                
                # Hough Line Transform (파라미터 강화)
                lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                                         minLineLength=40, maxLineGap=15)
                
                if lines is not None:
                    self.progress.emit(f"초기 검출: {len(lines)}개 직선", 82)
                    
                    # 1단계: 기반 필터링
                    temp_lines = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        dx = x2 - x1
                        dy = y2 - y1
                        
                        if abs(dx) < 5:  # 거의 수직선 제외
                            continue
                        
                        slope = dy / dx
                        length = np.sqrt(dx**2 + dy**2)
                        
                        # 빗변만: 기울기 0.5~3.0, 길이 50 이상
                        if 0.5 < abs(slope) < 3.0 and length > 50:
                            temp_lines.append({
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2),
                                'slope': float(slope),
                                'length': float(length)
                            })
                    
                    self.progress.emit(f"필터링 후: {len(temp_lines)}개", 84)
                    
                    # 2단계: 긴 직선 우선, 최대 50개로 제한
                    temp_lines.sort(key=lambda x: x['length'], reverse=True)
                    detected_lines = temp_lines[:50]  # 최대 50개
                
                # 상/하단 경계선과 교차점 계산
                height = z_projection.shape[0]
                width = z_projection.shape[1]
                top_y = int(height * 0.15)  # 상단 15%
                bottom_y = int(height * 0.85)  # 하단 85%
                
                self.progress.emit(f"교차점 계산 중... ({len(detected_lines)}개 직선)", 86)
                
                intersection_points = []  # 중복 확인용
                
                for line_info in detected_lines:
                    x1, y1, x2, y2 = line_info['x1'], line_info['y1'], line_info['x2'], line_info['y2']
                    slope = line_info['slope']
                    
                    # 직선 방정식: y = mx + c
                    c = y1 - slope * x1
                    
                    # 상승 빗변 → 상단 교차점
                    if slope > 0:
                        x_intersect = (top_y - c) / slope
                        y_intersect = top_y
                    # 하강 빗변 → 하단 교차점
                    else:
                        x_intersect = (bottom_y - c) / slope
                        y_intersect = bottom_y
                    
                    # 이미지 범위 내 + 중복 방지
                    if 0 <= x_intersect < width:
                        # 기존 교차점과 너무 가까우면 스킵
                        is_duplicate = any(abs(x_intersect - existing) < 20 for existing in intersection_points)
                        
                        if not is_duplicate:
                            line_info['intersection_x'] = float(x_intersect)
                            line_info['intersection_y'] = y_intersect
                            intersection_points.append(x_intersect)
                            
                            peak_positions.append({
                                'position': int(x_intersect) + 1,
                                'peak_strength': abs(slope) * line_info['length'],
                                'prominence': line_info['length'],
                                'y_position': y_intersect,
                                'slope': slope
                            })
                
                self.progress.emit(f"교차점 {len(intersection_points)}개 발견", 88)
                
                detection_params = {
                    'method': 'hough_line_intersection',
                    'top_boundary_y': top_y,
                    'bottom_boundary_y': bottom_y,
                    'num_lines_detected': len(detected_lines),
                    'num_intersections': len(peak_positions)
                }
        
        # 밝기 변화도 백업으로 사용
        brightness_changes = []
        for i in range(1, len(brightness_profile)):
            bright_diff = abs(brightness_profile[i]['brightness'] - brightness_profile[i-1]['brightness'])
            edge_diff = abs(brightness_profile[i]['edge_strength'] - brightness_profile[i-1]['edge_strength'])
            
            if bright_diff > 5 or edge_diff > 2:
                brightness_changes.append({
                    'position': i,
                    'brightness_change': bright_diff,
                    'edge_change': edge_diff,
                    'score': bright_diff + edge_diff * 2
                })
        
        # 높이 변화 감지
        height_changes = []
        for i in range(1, len(heights)):
            if heights[i]['height'] != heights[i-1]['height']:
                height_changes.append({
                    'position': i,
                    'from_height': heights[i-1]['height'],
                    'to_height': heights[i]['height'],
                    'change': heights[i]['height'] - heights[i-1]['height']
                })
        
        # 가장 많이 나타나는 높이 찾기
        all_heights = [h['height'] for h in heights]
        height_counter = Counter(all_heights)
        
        self.progress.emit("분석 완료!", 100)
        
        return {
            'folder': self.folder_path.name,
            'total_images': total,
            'heights': heights,
            'brightness_profile': brightness_profile,
            'brightness_changes': brightness_changes,
            'peak_positions': peak_positions,
            'height_changes': height_changes,
            'height_distribution': dict(height_counter),
            'z_projection': z_projection,
            'z_profile': z_profile,
            'image_files': image_files,
            'failed_files': failed_files,
            'detection_params': detection_params,
            'detected_lines': detected_lines
        }


class LayerBoundaryDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_folder = None
        self.analysis_result = None
        self.detected_layers = []
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Layer Boundary Detector - 층 경계 자동 탐지")
        self.setGeometry(100, 100, 1400, 900)
        
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        
        # 왼쪽 패널: 설정 및 로그
        left_panel = self.create_left_panel()
        
        # 오른쪽 패널: 그래프 및 결과
        right_panel = self.create_right_panel()
        
        # 스플리터
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        
        self.apply_style()
    
    def open_image_viewer(self):
        """이미지 뷰어 - 제거됨"""
        pass

    def create_left_panel(self):
        """왼쪽 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 폴더 선택
        folder_group = QGroupBox("폴더 선택")
        folder_layout = QVBoxLayout()
        
        folder_btn_layout = QHBoxLayout()
        self.folder_input = QLabel("폴더를 선택하세요")
        self.folder_input.setWordWrap(True)
        self.folder_input.setStyleSheet("QLabel { padding: 8px; background-color: #2a2a2a; border-radius: 3px; }")
        folder_btn_layout.addWidget(self.folder_input, 1)
        
        browse_btn = QPushButton("찾아보기")
        browse_btn.clicked.connect(self.browse_folder)
        folder_btn_layout.addWidget(browse_btn)
        
        folder_layout.addLayout(folder_btn_layout)
        
        self.analyze_btn = QPushButton("이미지 분석 시작")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(40)
        folder_layout.addWidget(self.analyze_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        folder_layout.addWidget(self.progress_bar)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # 층 설정
        layer_group = QGroupBox("층 설정")
        layer_layout = QGridLayout()
        
        layer_layout.addWidget(QLabel("총 층 수:"), 0, 0)
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 50)
        self.num_layers_spin.setValue(19)
        layer_layout.addWidget(self.num_layers_spin, 0, 1)
        
        layer_layout.addWidget(QLabel("예상 층당 이미지:"), 1, 0)
        self.images_per_layer_label = QLabel("~170장")
        layer_layout.addWidget(self.images_per_layer_label, 1, 1)
        
        self.auto_detect_btn = QPushButton("자동 경계 탐지")
        self.auto_detect_btn.clicked.connect(self.auto_detect_boundaries)
        self.auto_detect_btn.setEnabled(False)
        layer_layout.addWidget(self.auto_detect_btn, 2, 0, 1, 2)
        
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)
        
        # 저장
        save_group = QGroupBox("결과 저장")
        save_layout = QVBoxLayout()
        
        self.save_btn = QPushButton("layers.json으로 저장")
        self.save_btn.clicked.connect(self.save_layers)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # 로그
        log_group = QGroupBox("분석 로그")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self):
        """오른쪽 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 측면 이미지 (Z-Projection)
        zprojection_group = QGroupBox("측면 이미지 (Z-Projection) - Fiji Orthogonal View 스타일")
        zprojection_layout = QVBoxLayout()
        
        self.zprojection_figure, self.zprojection_ax = plt.subplots(figsize=(10, 3))
        self.zprojection_canvas = FigureCanvas(self.zprojection_figure)
        zprojection_layout.addWidget(self.zprojection_canvas)
        
        zprojection_group.setLayout(zprojection_layout)
        layout.addWidget(zprojection_group, 3)  # 크기 3배로 늘림
        
        # Hough Line 검출 상세 그래프
        hough_group = QGroupBox("Hough Line 검출 결과 (에지 + 직선 + 교차점)")
        hough_layout = QVBoxLayout()
        
        self.hough_figure, self.hough_ax = plt.subplots(figsize=(10, 3))
        self.hough_canvas = FigureCanvas(self.hough_figure)
        hough_layout.addWidget(self.hough_canvas)
        
        hough_group.setLayout(hough_layout)
        layout.addWidget(hough_group, 2)
        
        # 결과 테이블
        table_group = QGroupBox("탐지된 층 정보")
        table_layout = QVBoxLayout()
        
        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(4)
        self.layer_table.setHorizontalHeaderLabels(["층 번호", "시작", "끝", "이미지 수"])
        self.layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.layer_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group, 1)
        
        panel.setLayout(layout)
        return panel
    
    def browse_folder(self):
        """폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "분석할 폴더 선택")
        if folder:
            self.current_folder = Path(folder)
            self.folder_input.setText(str(self.current_folder))
            self.analyze_btn.setEnabled(True)
            self.log_text.append(f"폴더 선택: {self.current_folder.name}")
    
    def start_analysis(self):
        """이미지 분석 시작"""
        if not self.current_folder:
            return
        
        self.log_text.clear()
        self.log_text.append("=" * 60)
        self.log_text.append("이미지 분석 시작...")
        self.log_text.append("=" * 60)
        
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 분석 스레드 시작
        self.analyzer_thread = ImageAnalyzerThread(self.current_folder)
        self.analyzer_thread.progress.connect(self.on_analysis_progress)
        self.analyzer_thread.finished.connect(self.on_analysis_finished)
        self.analyzer_thread.error.connect(self.on_analysis_error)
        self.analyzer_thread.start()
    
    def on_analysis_progress(self, message, percent):
        """분석 진행 상황"""
        if percent >= 0:
            self.progress_bar.setValue(percent)
        self.log_text.append(message)
    
    def on_analysis_finished(self, result):
        """분석 완료"""
        self.analysis_result = result
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log_text.append(f"\n총 이미지: {result['total_images']}개")
        failed_count = len(result.get('failed_files', []))
        if failed_count > 0:
            self.log_text.append(f"읽기 실패: {failed_count}개")
            if failed_count <= 5:
                for failed in result['failed_files']:
                    self.log_text.append(f"  - {failed['name']}: {failed['error']}")
            else:
                self.log_text.append(f"  (처음 5개만 표시)")
                for failed in result['failed_files'][:5]:
                    self.log_text.append(f"  - {failed['name']}: {failed['error']}")
        
        # 탐지 파라미터 표시
        if result.get('detection_params'):
            params = result['detection_params']
            self.log_text.append(f"\n피크 탐지 파라미터:")
            self.log_text.append(f"  - 기울기 임계값: {params.get('slope_threshold', 0):.2f}")
            self.log_text.append(f"  - 최소 돌출도: {params.get('min_prominence', 0):.2f}")
            self.log_text.append(f"  - 윤활 윈도우: {params.get('window_size', 0)}")
        
        self.log_text.append(f"\n대각선 꼭지점 탐지: {len(result['peak_positions'])}개")
        if len(result['peak_positions']) > 0:
            peak_strengths = [p['peak_strength'] for p in result['peak_positions']]
            self.log_text.append(f"  피크 강도 범위: {min(peak_strengths):.1f} ~ {max(peak_strengths):.1f}")
        # 밝기/높이 변화 로그 제거됨
        
        # 높이 분포
        self.log_text.append("\n높이 분포:")
        for height, count in sorted(result['height_distribution'].items(), key=lambda x: -x[1])[:5]:
            self.log_text.append(f"  높이 {height}px: {count}개 이미지")
        
        # Z-Projection 정보
        if result.get('z_projection') is not None:
            z_shape = result['z_projection'].shape
            self.log_text.append(f"\n측면 이미지 생성: {z_shape[1]}x{z_shape[0]} pixels")
        
        # 그래프 그리기
        try:
            self.plot_zprojection()
        except Exception as e:
            self.log_text.append(f"\n[Z-Projection 그리기 오류] {str(e)}")
        
        try:
            self.plot_hough_detection()
        except Exception as e:
            self.log_text.append(f"\n[Hough 그리기 오류] {str(e)}")
        
        # 자동 탐지 버튼 활성화
        self.auto_detect_btn.setEnabled(True)
        
        # 예상 이미지 수 업데이트
        avg_per_layer = result['total_images'] / self.num_layers_spin.value()
        self.images_per_layer_label.setText(f"~{int(avg_per_layer)}장")
    
    def on_analysis_error(self, error_msg):
        """분석 오류"""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log_text.append(f"\n[오류] {error_msg}")
        QMessageBox.critical(self, "오류", f"분석 중 오류 발생:\n{error_msg}")
    
    def plot_hough_detection(self):
        """검출된 Hough Line 상세 시각화 (에지 + 직선 + 교차점)"""
        try:
            self.hough_ax.clear()
            
            if not self.analysis_result or self.analysis_result.get('z_projection') is None:
                self.hough_canvas.draw()
                return
            
            z_proj = self.analysis_result['z_projection']
            
            # 1. 원본 Z-projection 표시 (배경)
            self.hough_ax.imshow(z_proj, cmap='gray', aspect='auto', interpolation='bilinear', alpha=0.5)
            
            # 2. Canny 에지 오버레이
            z_proj_norm = cv2.normalize(z_proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            edges = cv2.Canny(z_proj_norm, 30, 100)
            self.hough_ax.imshow(edges, cmap='Reds', aspect='auto', alpha=0.3)
            
            # 3. 검출된 직선 그리기
            detected_lines = self.analysis_result.get('detected_lines', [])
            detection_params = self.analysis_result.get('detection_params', {})
            
            for line in detected_lines:
                x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
                slope = line['slope']
                
                if slope > 0:
                    color = 'lime'
                    label = '상승 빗변' if line == detected_lines[0] else ''
                else:
                    color = 'yellow'
                    label = '하강 빗변' if slope < 0 and label == '' else ''
                
                self.hough_ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.8, label=label)
                
                # 기울기 텍스트 표시
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                self.hough_ax.text(mid_x, mid_y, f'{slope:.2f}', fontsize=7, color='white', 
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            # 4. 상/하단 경계선
            if 'top_boundary_y' in detection_params:
                top_y = detection_params['top_boundary_y']
                bottom_y = detection_params['bottom_boundary_y']
                width = z_proj.shape[1]
                
                self.hough_ax.axhline(y=top_y, color='red', linestyle='-', linewidth=3, alpha=0.9, label=f'상단 (y={top_y})')
                self.hough_ax.axhline(y=bottom_y, color='red', linestyle='-', linewidth=3, alpha=0.9, label=f'하단 (y={bottom_y})')
            
            # 5. 교차점 표시
            intersection_count = 0
            for line in detected_lines:
                if 'intersection_x' in line:
                    x_int = line['intersection_x']
                    y_int = line['intersection_y']
                    self.hough_ax.scatter([x_int], [y_int], color='magenta', s=150, 
                                         marker='X', edgecolors='white', linewidths=2, 
                                         zorder=10, alpha=1.0)
                    # 교차점 x좌표 텍스트
                    self.hough_ax.text(x_int, y_int - 15, f'x={int(x_int)}', fontsize=8, 
                                      color='magenta', fontweight='bold', ha='center',
                                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
                    intersection_count += 1
            
            self.hough_ax.set_xlabel('Image Index', fontsize=11)
            self.hough_ax.set_ylabel('Height (pixel)', fontsize=11)
            title = f'Hough Line 검출: {len(detected_lines)}개 직선, {intersection_count}개 교차점'
            self.hough_ax.set_title(title, fontsize=11, fontweight='bold')
            self.hough_ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            self.hough_ax.grid(False)
            
            self.hough_canvas.draw()
        except Exception as e:
            self.log_text.append(f"Hough 시각화 오류: {str(e)}")
            import traceback
            self.log_text.append(traceback.format_exc())
    
    def plot_zprojection(self):
        """측면 이미지 (Z-Projection) + 검출된 직선 및 교차점 표시"""
        try:
            if not self.analysis_result or self.analysis_result.get('z_projection') is None:
                return
            
            self.zprojection_ax.clear()
            
            z_proj = self.analysis_result['z_projection']
            
            # 이미지 표시
            self.zprojection_ax.imshow(z_proj, cmap='gray', aspect='auto', interpolation='bilinear')
            
            # 검출된 빗변 직선 그리기
            detected_lines = self.analysis_result.get('detected_lines', [])
            for line in detected_lines:
                x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
                slope = line['slope']
                color = 'cyan' if slope > 0 else 'yellow'
                self.zprojection_ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.7)
            
            # 상/하단 경계선
            detection_params = self.analysis_result.get('detection_params', {})
            if 'top_boundary_y' in detection_params:
                top_y = detection_params['top_boundary_y']
                bottom_y = detection_params['bottom_boundary_y']
                self.zprojection_ax.axhline(y=top_y, color='red', linestyle='--', linewidth=2, alpha=0.8)
                self.zprojection_ax.axhline(y=bottom_y, color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            # 교차점
            for line in detected_lines:
                if 'intersection_x' in line:
                    x_int = line['intersection_x']
                    y_int = line['intersection_y']
                    self.zprojection_ax.scatter([x_int], [y_int], color='magenta', s=100, 
                                               marker='o', edgecolors='white', linewidths=2, zorder=5)
            
            self.zprojection_ax.set_xlabel('Image Index (권취 방향)')
            self.zprojection_ax.set_ylabel('Height (픽셀)')
            title = '측면 이미지 (Z-Projection)'
            if detected_lines:
                title += f' - {len(detected_lines)}개 직선 검출'
            self.zprojection_ax.set_title(title)
            
            self.zprojection_canvas.draw()
        except Exception as e:
            self.log_text.append(f"Z-Projection 시각화 오류: {str(e)}")
            import traceback
            self.log_text.append(traceback.format_exc())
    
    def auto_detect_boundaries(self):
        """자동 경계 탐지"""
        if not self.analysis_result:
            return
        
        self.log_text.append("\n" + "=" * 60)
        self.log_text.append("자동 경계 탐지 시작...")
        
        num_layers = self.num_layers_spin.value()
        total_images = self.analysis_result['total_images']
        images_per_layer = total_images / num_layers if num_layers else total_images
        
        # 방법 1: 대각선 꼭지점 기반 (최우선)
        peak_positions = self.analysis_result.get('peak_positions', [])
        boundary_positions = []
        
        if peak_positions:
            self.log_text.append(f"Z-Projection 삼각형 꼭지점 탐지: {len(peak_positions)}개")
            
            # 탐지된 모든 피크를 위치 순으로 정렬
            sorted_peaks = sorted(peak_positions, key=lambda x: x['position'])
            
            # 로그에 모든 피크 위치 표시
            all_peak_positions = [p['position'] for p in sorted_peaks]
            self.log_text.append(f"피크 위치: {all_peak_positions}")
            
            # 필요한 경계 수
            required_boundaries = num_layers - 1
            
            # 너무 가까운 피크 제거 (최소 간격)
            min_distance = max(30, int(images_per_layer * 0.3)) if images_per_layer else 30
            filtered_peaks = []
            for peak in sorted_peaks:
                pos = peak['position']
                # 첫 번째이거나, 이전 피크와 충분히 떨어져 있으면 추가
                if not filtered_peaks or (pos - filtered_peaks[-1]['position']) >= min_distance:
                    filtered_peaks.append(peak)
            
            self.log_text.append(f"필터링 후: {len(filtered_peaks)}개 (최소간격: {min_distance})")
            
            # 가장 강한 피크들을 선택
            if len(filtered_peaks) > required_boundaries:
                # prominence 기준으로 정렬하여 상위 N개 선택
                selected = sorted(filtered_peaks, key=lambda x: x['prominence'], reverse=True)[:required_boundaries]
                # 다시 위치 순으로 정렬
                boundary_positions = sorted([p['position'] for p in selected])
            else:
                boundary_positions = [p['position'] for p in filtered_peaks]
            
            self.log_text.append(f"최종 선택된 경계: {boundary_positions}")
            
        else:
            self.log_text.append("꼭지점을 찾을 수 없습니다")
            boundary_positions = []
        
        # 경계가 충분하지 않으면 균등 분할로 보완
        if len(boundary_positions) < num_layers - 1:
            self.log_text.append(f"경고: {len(boundary_positions)}개 경계만 탐지됨, 균등 분할로 보완")
            images_per_layer = total_images // num_layers
            for i in range(num_layers - 1):
                expected_pos = (i + 1) * images_per_layer
                # 기존 경계와 겹치지 않으면 추가
                if not any(abs(expected_pos - existing) < 50 for existing in boundary_positions):
                    boundary_positions.append(expected_pos)
            boundary_positions = sorted(boundary_positions)[:num_layers-1]
        
        # 층 정보 생성
        self.detected_layers = []
        prev_pos = 1
        
        for layer_id, boundary in enumerate(boundary_positions, start=1):
            self.detected_layers.append({
                'layer': layer_id,
                'start': prev_pos,
                'end': boundary,
                'count': boundary - prev_pos + 1
            })
            prev_pos = boundary + 1
        
        # 마지막 층
        self.detected_layers.append({
            'layer': num_layers,
            'start': prev_pos,
            'end': total_images,
            'count': total_images - prev_pos + 1
        })
        
        # 결과 표시
        self.display_layer_table()
        self.plot_detected_layers()
        
        self.log_text.append(f"{num_layers}개 층 탐지 완료!")
        self.save_btn.setEnabled(True)
    
    def display_layer_table(self):
        """층 정보 테이블 표시"""
        self.layer_table.setRowCount(len(self.detected_layers))
        
        for row, layer in enumerate(self.detected_layers):
            self.layer_table.setItem(row, 0, QTableWidgetItem(str(layer['layer'])))
            self.layer_table.setItem(row, 1, QTableWidgetItem(str(layer['start'])))
            self.layer_table.setItem(row, 2, QTableWidgetItem(str(layer['end'])))
            self.layer_table.setItem(row, 3, QTableWidgetItem(str(layer['count'])))
    
    def plot_detected_layers(self):
        """탐지된 층 경계 그래프 및 측면 이미지에 표시"""
        # 측면 이미지에 층 경계 표시
        if self.analysis_result.get('z_projection') is not None:
            self.zprojection_ax.clear()
            z_proj = self.analysis_result['z_projection']
            self.zprojection_ax.imshow(z_proj, cmap='gray', aspect='auto', interpolation='bilinear')
            
            # 층 경계선 추가 (빨간 수직선)
            for layer in self.detected_layers[:-1]:
                self.zprojection_ax.axvline(x=layer['end'], color='red', linewidth=2, alpha=0.8)
                # 층 번호 표시
                self.zprojection_ax.text(layer['end'], 10, f"L{layer['layer']}", 
                                        color='yellow', fontsize=8, fontweight='bold')
            
            self.zprojection_ax.set_xlabel('Image Index (권취 방향)')
            self.zprojection_ax.set_ylabel('Height (픽셀)')
            self.zprojection_ax.set_title('측면 이미지 - 빨간 선: 탐지된 층 경계')
            self.zprojection_canvas.draw()
    
    def save_layers(self):
        """layers.json 저장"""
        if not self.detected_layers:
            return
        
        # 저장 위치 선택
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "layers.json 저장",
            str(Path.cwd() / f"layers_{self.current_folder.name}.json"),
            "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        # JSON 구조 생성
        layers_data = {
            self.current_folder.name: self.detected_layers
        }
        
        # 저장
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(layers_data, f, indent=2, ensure_ascii=False)
            
            self.log_text.append(f"\n저장 완료: {file_path}")
            QMessageBox.information(
                self,
                "저장 완료",
                f"layers.json 파일이 저장되었습니다:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "오류", f"저장 중 오류:\n{str(e)}")
    
    def apply_style(self):
        """스타일 적용"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QGroupBox {
                border: 2px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
            QLineEdit, QSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QTextEdit, QTableWidget {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
            }
            QLabel {
                color: #d4d4d4;
            }
        """)


# ImageViewerDialog 클래스 제거됨


def main():
    app = QApplication(sys.argv)
    window = LayerBoundaryDetector()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
