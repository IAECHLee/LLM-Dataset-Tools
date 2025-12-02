#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
권취 모델 테스트 GUI - 이미지 분류 모델 검증 도구

기능:
1. 폴더 내 전체 이미지에 대해 모델 추론 실행
2. 이미지별 예측 결과 확인
3. 분류 통계 및 혼동 행렬 표시
"""

import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QSplitter,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGroupBox,
    QMessageBox, QShortcut, QFrame, QFileDialog, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QTabWidget,
    QSpinBox
)
from PyQt5.QtCore import Qt, QRectF, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QKeySequence, QFont, QColor, QPainter, QImage
import nrt


class ImageLoaderThread(QThread):
    """비동기 이미지 로더"""
    loaded = pyqtSignal(QPixmap, str)  # pixmap, filepath
    
    def __init__(self, filepath, max_size=1920):
        super().__init__()
        self.filepath = filepath
        self.max_size = max_size
    
    def run(self):
        try:
            # OpenCV로 빠르게 로드
            img = cv2.imread(self.filepath)
            if img is None:
                return
            
            # 큰 이미지는 리사이즈 (메모리 및 속도 최적화)
            h, w = img.shape[:2]
            if max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            
            # QPixmap으로 변환
            qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg.copy())  # copy() 중요: 메모리 안전
            
            self.loaded.emit(pixmap, self.filepath)
        except Exception as e:
            print(f"이미지 로드 오류: {e}")


class ZoomableGraphicsView(QGraphicsView):
    """줌 및 패닝 지원 이미지 뷰어 (비동기 로딩 + 캐싱)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setFrameShape(QFrame.NoFrame)
        
        # 캐싱 활성화
        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setOptimizationFlags(QGraphicsView.DontAdjustForAntialiasing)
        
        self._zoom = 1.0
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        
        # 비동기 로더
        self._loader_thread = None
        self._current_path = None
        
        # 이미지 캐시 (LRU)
        self._image_cache = {}
        self._cache_order = []
        self._max_cache = 30  # 최대 30개 캐싱
    
    def set_image(self, image_path):
        """이미지 로드 및 표시 (동기 방식으로 변경 - 안정성)"""
        if not image_path or not os.path.exists(image_path):
            return False
        
        self._current_path = image_path
        
        # 캐시 확인
        if image_path in self._image_cache:
            self._display_pixmap(self._image_cache[image_path])
            # LRU 업데이트
            if image_path in self._cache_order:
                self._cache_order.remove(image_path)
            self._cache_order.append(image_path)
            return True
        
        # 동기 로드 (안정성 우선)
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"이미지 로드 실패: {image_path}")
                return False
            
            # 큰 이미지 리사이즈
            h, w = img.shape[:2]
            max_size = 1920
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            
            # QPixmap으로 변환
            qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg.copy())
            
            # 캐시에 저장
            self._image_cache[image_path] = pixmap
            self._cache_order.append(image_path)
            
            # 캐시 크기 제한
            while len(self._cache_order) > self._max_cache:
                old_path = self._cache_order.pop(0)
                if old_path in self._image_cache:
                    del self._image_cache[old_path]
            
            self._display_pixmap(pixmap)
            return True
            
        except Exception as e:
            print(f"이미지 로드 오류: {e}")
            return False
    
    def _on_image_loaded(self, pixmap, filepath):
        """이미지 로드 완료 콜백"""
        # 현재 선택된 이미지와 일치하는지 확인
        if filepath != self._current_path:
            return
        
        # 캐시에 저장
        self._image_cache[filepath] = pixmap
        self._cache_order.append(filepath)
        
        # 캐시 크기 제한
        while len(self._cache_order) > self._max_cache:
            old_path = self._cache_order.pop(0)
            if old_path in self._image_cache:
                del self._image_cache[old_path]
        
        self._display_pixmap(pixmap)
    
    def _display_pixmap(self, pixmap):
        """픽스맵 표시"""
        self._scene.clear()
        if not pixmap.isNull():
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self._pixmap_item)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.fit_in_view()
    
    def fit_in_view(self):
        """이미지를 뷰에 맞춤"""
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._zoom = 1.0
    
    def wheelEvent(self, event):
        """마우스 휠로 줌"""
        factor = 1.2
        if event.angleDelta().y() > 0:
            self._zoom *= factor
            self.scale(factor, factor)
        else:
            self._zoom /= factor
            self.scale(1/factor, 1/factor)
    
    def reset_view(self):
        """뷰 리셋"""
        self.resetTransform()
        self.fit_in_view()
    
    def mouseDoubleClickEvent(self, event):
        """더블클릭으로 뷰 리셋"""
        self.reset_view()
        super().mouseDoubleClickEvent(event)
    
    def clear_cache(self):
        """캐시 비우기"""
        self._image_cache.clear()
        self._cache_order.clear()


class InferenceThread(QThread):
    """백그라운드 추론 스레드 (배치 처리 지원)"""
    progress = pyqtSignal(int, int, str)  # current, total, filename
    preview = pyqtSignal(object, str, str, float)  # image_array, filepath, predicted_class, confidence
    batch_results = pyqtSignal(list)  # 배치 결과 한번에 전송
    finished = pyqtSignal(float, int, list)  # elapsed_time, total_images, all_results
    error = pyqtSignal(str)
    
    def __init__(self, model_path, image_folder, use_gpu=True, batch_size=8):
        super().__init__()
        self.model_path = model_path
        self.image_folder = Path(image_folder)
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._stop = False
    
    def stop(self):
        self._stop = True
    
    def run(self):
        try:
            start_time = time.time()  # 시작 시간 측정
            
            # Predictor 생성 (NRT 공식 API 사용)
            # GPU: device_idx = 0, CPU: device_idx = -1
            device_idx = 0 if self.use_gpu else -1
            fp16_flag = False
            threshold_flag = False
            
            if device_idx >= 0:
                # GPU 모드
                predictor = nrt.Predictor(
                    str(self.model_path),
                    nrt.Model.MODELIO_DEFAULT,
                    device_idx,
                    self.batch_size,
                    fp16_flag,
                    threshold_flag,
                    nrt.DEVICE_CUDA_GPU
                )
            else:
                # CPU 모드
                predictor = nrt.Predictor(
                    str(self.model_path),
                    nrt.Model.MODELIO_DEFAULT,
                    device_idx,
                    self.batch_size,
                    fp16_flag,
                    threshold_flag
                )
            
            if predictor.get_status() != nrt.STATUS_SUCCESS:
                raise Exception("Predictor 초기화 실패: " + nrt.get_last_error_msg())
            
            # 클래스 정보
            num_classes = predictor.get_num_classes()
            
            # 이미지 파일 목록
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            image_files = []
            for f in sorted(self.image_folder.iterdir()):
                if f.suffix.lower() in image_extensions:
                    image_files.append(f)
            
            total = len(image_files)
            processed = 0
            
            # 모든 결과를 저장할 리스트
            all_results = []
            preview_interval = 10  # 10장마다 미리보기
            
            # 배치 단위로 처리
            for batch_start in range(0, total, self.batch_size):
                if self._stop:
                    break
                
                batch_end = min(batch_start + self.batch_size, total)
                batch_files = image_files[batch_start:batch_end]
                
                # progress는 매 배치마다 emit (간략히)
                self.progress.emit(batch_end, total, f"처리 중... {batch_end}/{total}")
                
                try:
                    # Input 생성 및 배치 이미지 추가
                    inputs = nrt.Input()
                    valid_files = []
                    
                    for img_path in batch_files:
                        status = inputs.extend(str(img_path))
                        if status == nrt.STATUS_SUCCESS:
                            valid_files.append(img_path)
                        else:
                            print(f"입력 추가 실패: {img_path}")
                    
                    if not valid_files:
                        continue
                    
                    # 배치 추론
                    results = predictor.predict(inputs)
                    
                    if results.get_status() != nrt.STATUS_SUCCESS:
                        print(f"배치 추론 실패")
                        continue
                    
                    # 배치 결과 파싱
                    for batch_idx, img_path in enumerate(valid_files):
                        if self._stop:
                            break
                        
                        try:
                            # 해당 배치 인덱스의 결과 가져오기
                            top_class = results.classes.get(batch_idx)
                            class_idx = top_class.idx
                            predicted_class = predictor.get_class_name(class_idx)
                            confidence = results.probs.get(batch_idx, class_idx)
                            
                            # 모든 클래스 확률
                            all_probs = []
                            for j in range(num_classes):
                                class_name = predictor.get_class_name(j)
                                prob = results.probs.get(batch_idx, j)
                                all_probs.append((class_name, prob))
                            
                            # 확률 순으로 정렬
                            all_probs.sort(key=lambda x: x[1], reverse=True)
                            
                            # 결과 저장
                            all_results.append((str(img_path), predicted_class, confidence, all_probs))
                            processed += 1
                            
                            # 10장마다 미리보기 업데이트 (이미지 포함)
                            if processed % preview_interval == 0:
                                # 미리보기용 이미지 로드 (RGB)
                                preview_img = cv2.imread(str(img_path))
                                if preview_img is not None:
                                    # 큰 이미지 리사이즈
                                    h, w = preview_img.shape[:2]
                                    max_size = 1280
                                    if max(h, w) > max_size:
                                        scale = max_size / max(h, w)
                                        preview_img = cv2.resize(preview_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                                    preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                                    self.preview.emit(preview_img, str(img_path), predicted_class, confidence)
                            
                        except Exception as e:
                            print(f"결과 파싱 오류 {img_path}: {e}")
                            continue
                    
                except Exception as e:
                    print(f"배치 처리 오류: {e}")
                    continue
            
            elapsed_time = time.time() - start_time  # 소요 시간 계산
            self.finished.emit(elapsed_time, total, all_results)  # 전체 결과 함께 전송
            
        except Exception as e:
            self.error.emit(str(e))


class HeatmapThread(QThread):
    """히트맵(CAM) 생성 스레드"""
    finished = pyqtSignal(np.ndarray, str)  # heatmap_image, filepath
    error = pyqtSignal(str)
    
    def __init__(self, model_path, image_path, use_gpu=True):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.use_gpu = use_gpu
    
    def run(self):
        try:
            device_idx = 0 if self.use_gpu else -1
            
            # CAM 출력 활성화된 Predictor 생성
            if device_idx >= 0:
                predictor = nrt.Predictor(
                    str(self.model_path),
                    nrt.Model.MODELIO_OUT_CAM,  # CAM 출력 활성화
                    device_idx,
                    1,  # batch_size = 1
                    False,
                    False,
                    nrt.DEVICE_CUDA_GPU
                )
            else:
                predictor = nrt.Predictor(
                    str(self.model_path),
                    nrt.Model.MODELIO_OUT_CAM,
                    device_idx,
                    1,
                    False,
                    False
                )
            
            if predictor.get_status() != nrt.STATUS_SUCCESS:
                raise Exception("Predictor 초기화 실패: " + nrt.get_last_error_msg())
            
            # 입력 이미지
            inputs = nrt.Input()
            status = inputs.extend(str(self.image_path))
            if status != nrt.STATUS_SUCCESS:
                raise Exception("입력 이미지 로드 실패")
            
            # 추론 (CAM 포함)
            results = predictor.predict(inputs)
            
            if results.get_status() != nrt.STATUS_SUCCESS:
                raise Exception("추론 실패: " + nrt.get_last_error_msg())
            
            # CAM 추출
            if not results.cams.empty():
                cam = results.cams.get(0)
                mat_cam = cam.cam_to_numpy()
                mat_cam = mat_cam.reshape([cam.get_height(), cam.get_width(), 3])
                
                # 원본 이미지 로드
                original = cv2.imread(str(self.image_path))
                if original is not None:
                    # CAM을 원본 크기로 리사이즈
                    cam_resized = cv2.resize(mat_cam, (original.shape[1], original.shape[0]))
                    
                    # 원본 이미지와 히트맵 블렌딩
                    blended = cv2.addWeighted(original, 0.6, cam_resized, 0.4, 0)
                    
                    # BGR to RGB
                    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                    
                    self.finished.emit(blended, str(self.image_path))
                else:
                    self.finished.emit(mat_cam, str(self.image_path))
            else:
                raise Exception("CAM 데이터가 없습니다. 모델이 CAM을 지원하지 않을 수 있습니다.")
                
        except Exception as e:
            self.error.emit(str(e))


class ModelTestGUI(QMainWindow):
    """모델 테스트 메인 GUI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("권취 모델 테스트 - 이미지 분류 검증")
        self.setGeometry(100, 100, 1600, 900)
        
        # 모델 폴더 및 경로
        self.model_folder = Path(r"D:\LLM_Dataset\models")
        self.model_path = None
        self.image_folder = None
        
        # 추론 결과 저장
        self.results = {}  # {filepath: (predicted_class, confidence, all_probs)}
        self.class_names = []
        
        # UI 업데이트 타이머 (예비용)
        self.ui_update_timer = QTimer()
        
        # 스레드
        self.inference_thread = None
        
        # 히트맵 관련
        self.heatmap_thread = None
        self.current_selected_path = None
        self.heatmap_cache = {}  # {filepath: heatmap_pixmap}
        self.showing_heatmap = False
        
        self.init_ui()
        self.setup_shortcuts()
        self.load_models()  # 모델 목록 로드
    
    def init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 상단: 폴더 선택 및 실행
        top_layout = QHBoxLayout()
        
        top_layout.addWidget(QLabel("테스트 폴더:"))
        self.folder_label = QLabel("폴더를 선택하세요")
        self.folder_label.setStyleSheet("QLabel { color: #888; padding: 5px; background-color: #2a2a2a; border-radius: 3px; }")
        self.folder_label.setMinimumWidth(400)
        top_layout.addWidget(self.folder_label)
        
        browse_btn = QPushButton("📁 폴더 선택")
        browse_btn.clicked.connect(self.browse_folder)
        top_layout.addWidget(browse_btn)
        
        self.run_btn = QPushButton("▶ 추론 시작")
        self.run_btn.clicked.connect(self.start_inference)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #0a7a0a;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #0c9a0c;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
            }
        """)
        top_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("⏹ 중지")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #a00000;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
            }
        """)
        top_layout.addWidget(self.stop_btn)
        
        # 배치 사이즈 설정
        top_layout.addWidget(QLabel("배치 크기:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        self.batch_spin.setToolTip("배치 크기가 클수록 빠르지만 GPU 메모리를 더 많이 사용합니다")
        self.batch_spin.setStyleSheet("""
            QSpinBox {
                background-color: #2a2a2a;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 3px 8px;
                min-width: 60px;
            }
            QSpinBox:hover {
                border-color: #4a9eff;
            }
        """)
        top_layout.addWidget(self.batch_spin)
        
        top_layout.addStretch()
        
        # 모델 선택
        top_layout.addWidget(QLabel("모델:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 5px 10px;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #4a9eff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                border: 1px solid #3c3c3c;
                selection-background-color: #4a9eff;
            }
        """)
        top_layout.addWidget(self.model_combo)
        
        # 모델 폴더 선택 버튼
        model_browse_btn = QPushButton("📂")
        model_browse_btn.setToolTip("모델 폴더 변경")
        model_browse_btn.clicked.connect(self.browse_model_folder)
        model_browse_btn.setFixedWidth(40)
        top_layout.addWidget(model_browse_btn)
        
        # 모델 정보 레이블
        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        top_layout.addWidget(self.model_info_label)
        
        main_layout.addLayout(top_layout)
        
        # 진행 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 탭 위젯
        tab_widget = QTabWidget()
        
        # 탭 1: 이미지 뷰어
        viewer_tab = self.create_viewer_tab()
        tab_widget.addTab(viewer_tab, "🖼 이미지 뷰어")
        
        # 탭 2: 통계
        stats_tab = self.create_stats_tab()
        tab_widget.addTab(stats_tab, "📊 통계")
        
        main_layout.addWidget(tab_widget)
        
        # 다크 테마 스타일
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
            QPushButton:pressed {
                background-color: #0d5689;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                color: #d4d4d4;
            }
            QListWidget::item:selected {
                background-color: #0e639c;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #2a2a2a;
            }
            QLabel {
                color: #d4d4d4;
            }
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #d4d4d4;
                padding: 8px 20px;
                border: 1px solid #3c3c3c;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
            }
            QTableWidget {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                gridline-color: #3c3c3c;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #0e639c;
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                padding: 5px;
                border: 1px solid #3c3c3c;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px 10px;
                border-radius: 3px;
                color: #d4d4d4;
            }
        """)
    
    def create_viewer_tab(self):
        """이미지 뷰어 탭 생성"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # 왼쪽: 결과 리스트
        left_panel = QGroupBox("추론 결과")
        left_layout = QVBoxLayout(left_panel)
        
        # 필터
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("필터:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["전체", "Normal", "Twist", "Hook"])
        self.filter_combo.currentTextChanged.connect(self.filter_results)
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()
        left_layout.addLayout(filter_layout)
        
        # 진행 상황 레이블
        self.result_count_label = QLabel("0 / 0")
        self.result_count_label.setAlignment(Qt.AlignCenter)
        self.result_count_label.setStyleSheet("QLabel { font-weight: bold; color: #4a9eff; padding: 5px; }")
        left_layout.addWidget(self.result_count_label)
        
        # 결과 리스트 (성능 최적화)
        self.result_list = QListWidget()
        self.result_list.setFont(QFont("Consolas", 9))
        self.result_list.setUniformItemSizes(True)  # 동일 크기 아이템 - 성능 향상
        self.result_list.setLayoutMode(QListWidget.Batched)  # 배치 레이아웃
        self.result_list.setBatchSize(50)  # 50개씩 배치
        self.result_list.currentRowChanged.connect(self.on_result_selected)
        left_layout.addWidget(self.result_list)
        
        # 오른쪽: 이미지 뷰어
        right_panel = QGroupBox("이미지 미리보기")
        right_layout = QVBoxLayout(right_panel)
        
        # 예측 결과 표시
        self.prediction_label = QLabel("이미지를 선택하세요")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.prediction_label.setStyleSheet("QLabel { color: #888; padding: 10px; background-color: #2a2a2a; border-radius: 5px; }")
        self.prediction_label.setMinimumHeight(60)
        right_layout.addWidget(self.prediction_label)
        
        # 확률 표시
        self.prob_label = QLabel("")
        self.prob_label.setAlignment(Qt.AlignCenter)
        self.prob_label.setStyleSheet("QLabel { color: #888; padding: 5px; }")
        right_layout.addWidget(self.prob_label)
        
        # 히트맵 버튼
        heatmap_layout = QHBoxLayout()
        self.heatmap_btn = QPushButton("🔥 히트맵 생성")
        self.heatmap_btn.setEnabled(False)
        self.heatmap_btn.clicked.connect(self.generate_heatmap)
        self.heatmap_btn.setStyleSheet("""
            QPushButton {
                background-color: #6a3093;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
            }
        """)
        heatmap_layout.addWidget(self.heatmap_btn)
        
        self.show_original_btn = QPushButton("🖼 원본 보기")
        self.show_original_btn.setEnabled(False)
        self.show_original_btn.clicked.connect(self.show_original_image)
        self.show_original_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
            }
        """)
        heatmap_layout.addWidget(self.show_original_btn)
        
        heatmap_layout.addStretch()
        
        self.heatmap_status_label = QLabel("")
        self.heatmap_status_label.setStyleSheet("QLabel { color: #888; font-size: 9pt; }")
        heatmap_layout.addWidget(self.heatmap_status_label)
        
        right_layout.addLayout(heatmap_layout)
        
        # 이미지 뷰어
        self.image_viewer = ZoomableGraphicsView()
        right_layout.addWidget(self.image_viewer)
        
        # 파일 정보
        self.file_info_label = QLabel("")
        self.file_info_label.setAlignment(Qt.AlignCenter)
        self.file_info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; padding: 5px; }")
        right_layout.addWidget(self.file_info_label)
        
        # 스플리터
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        layout.addWidget(splitter)
        
        return widget
    
    def create_stats_tab(self):
        """통계 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 요약 통계
        summary_group = QGroupBox("분류 요약")
        summary_layout = QHBoxLayout(summary_group)
        
        self.total_label = QLabel("전체: 0")
        self.total_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.total_label.setStyleSheet("QLabel { color: #4a9eff; padding: 10px; }")
        summary_layout.addWidget(self.total_label)
        
        self.normal_count_label = QLabel("Normal: 0 (0.0%)")
        self.normal_count_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.normal_count_label.setStyleSheet("QLabel { color: #4caf50; padding: 10px; }")
        summary_layout.addWidget(self.normal_count_label)
        
        self.twist_count_label = QLabel("Twist: 0 (0.0%)")
        self.twist_count_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.twist_count_label.setStyleSheet("QLabel { color: #ff9800; padding: 10px; }")
        summary_layout.addWidget(self.twist_count_label)
        
        self.hook_count_label = QLabel("Hook: 0 (0.0%)")
        self.hook_count_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.hook_count_label.setStyleSheet("QLabel { color: #f44336; padding: 10px; }")
        summary_layout.addWidget(self.hook_count_label)
        
        summary_layout.addStretch()
        
        layout.addWidget(summary_group)
        
        # 신뢰도 통계
        confidence_group = QGroupBox("신뢰도 통계")
        confidence_layout = QHBoxLayout(confidence_group)
        
        self.avg_conf_label = QLabel("평균 신뢰도: -")
        self.avg_conf_label.setFont(QFont("Arial", 11))
        confidence_layout.addWidget(self.avg_conf_label)
        
        self.min_conf_label = QLabel("최소 신뢰도: -")
        self.min_conf_label.setFont(QFont("Arial", 11))
        confidence_layout.addWidget(self.min_conf_label)
        
        self.max_conf_label = QLabel("최대 신뢰도: -")
        self.max_conf_label.setFont(QFont("Arial", 11))
        confidence_layout.addWidget(self.max_conf_label)
        
        confidence_layout.addStretch()
        
        layout.addWidget(confidence_group)
        
        # 클래스별 상세 테이블
        table_group = QGroupBox("클래스별 상세 통계")
        table_layout = QVBoxLayout(table_group)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels(["클래스", "개수", "비율 (%)", "평균 신뢰도", "최소 신뢰도"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setRowCount(3)
        table_layout.addWidget(self.stats_table)
        
        layout.addWidget(table_group)
        
        # 낮은 신뢰도 이미지 목록
        low_conf_group = QGroupBox("낮은 신뢰도 이미지 (< 80%)")
        low_conf_layout = QVBoxLayout(low_conf_group)
        
        self.low_conf_list = QListWidget()
        self.low_conf_list.setFont(QFont("Consolas", 9))
        self.low_conf_list.itemDoubleClicked.connect(self.on_low_conf_item_clicked)
        low_conf_layout.addWidget(self.low_conf_list)
        
        layout.addWidget(low_conf_group)
        
        return widget
    
    def load_models(self):
        """모델 폴더에서 .net 파일 목록 로드"""
        self.model_combo.clear()
        
        if self.model_folder.exists():
            net_files = sorted(self.model_folder.glob("*.net"))
            for net_file in net_files:
                self.model_combo.addItem(net_file.name, str(net_file))
        
        if self.model_combo.count() == 0:
            self.model_combo.addItem("모델 없음")
            self.run_btn.setEnabled(False)
        else:
            # 첫 번째 모델 선택
            self.model_combo.setCurrentIndex(0)
    
    def on_model_changed(self, model_name):
        """모델 선택 변경"""
        if model_name == "모델 없음":
            self.model_path = None
            self.model_info_label.setText("")
            self.class_names = []
            return
        
        model_path = self.model_combo.currentData()
        if model_path:
            self.model_path = Path(model_path)
            self.load_model_info()
    
    def load_model_info(self):
        """선택된 모델 정보 로드"""
        if not self.model_path or not self.model_path.exists():
            return
        
        try:
            # NRT 모델 정보 로드
            model = nrt.Model(str(self.model_path), True)
            num_classes = model.get_num_classes()
            
            # 클래스 이름 업데이트
            self.class_names = []
            for i in range(num_classes):
                self.class_names.append(model.get_class_name(i))
            
            # 모델 정보 표시
            model_type = model.get_model_type()
            self.model_info_label.setText(f"({num_classes}개 클래스: {', '.join(self.class_names)})")
            
            # 통계 테이블 행 수 업데이트
            self.stats_table.setRowCount(num_classes)
            
        except Exception as e:
            self.model_info_label.setText(f"(로드 실패: {str(e)[:30]})")
            self.class_names = []
    
    def browse_model_folder(self):
        """모델 폴더 변경"""
        folder = QFileDialog.getExistingDirectory(
            self, "모델 폴더 선택",
            str(self.model_folder)
        )
        if folder:
            self.model_folder = Path(folder)
            self.load_models()
    
    def setup_shortcuts(self):
        """단축키 설정"""
        QShortcut(QKeySequence(Qt.Key_Up), self, self.select_previous)
        QShortcut(QKeySequence(Qt.Key_Down), self, self.select_next)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.image_viewer.reset_view)
    
    def browse_folder(self):
        """폴더 선택"""
        folder = QFileDialog.getExistingDirectory(
            self, "테스트 폴더 선택", 
            str(Path(r"K:\LLM Image_Storage"))
        )
        if folder:
            self.image_folder = Path(folder)
            self.folder_label.setText(str(self.image_folder))
            self.folder_label.setStyleSheet("QLabel { color: #4a9eff; padding: 5px; background-color: #2a2a2a; border-radius: 3px; }")
            self.run_btn.setEnabled(True)
            
            # 이미지 개수 확인
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            count = sum(1 for f in self.image_folder.iterdir() if f.suffix.lower() in image_extensions)
            self.result_count_label.setText(f"이미지: {count}개")
    
    def start_inference(self):
        """추론 시작"""
        if not self.image_folder:
            QMessageBox.warning(self, "경고", "테스트 폴더를 선택하세요.")
            return
        
        if not self.model_path or not self.model_path.exists():
            QMessageBox.warning(self, "경고", "모델을 선택하세요.")
            return
        
        # 초기화
        self.results.clear()
        self.result_list.clear()
        self.low_conf_list.clear()
        
        # 이미지 캐시 비우기
        self.image_viewer.clear_cache()
        
        # 히트맵 캐시 비우기
        self.heatmap_cache.clear()
        self.current_selected_path = None
        self.heatmap_btn.setEnabled(False)
        self.show_original_btn.setEnabled(False)
        self.heatmap_status_label.setText("")
        
        # UI 상태 변경
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 스레드 시작
        self.inference_thread = InferenceThread(
            str(self.model_path),
            str(self.image_folder),
            use_gpu=True,
            batch_size=self.batch_spin.value()
        )
        self.inference_thread.progress.connect(self.on_progress, Qt.QueuedConnection)
        self.inference_thread.preview.connect(self.on_preview, Qt.QueuedConnection)  # 10장마다 미리보기
        self.inference_thread.finished.connect(self.on_finished, Qt.QueuedConnection)
        self.inference_thread.error.connect(self.on_error, Qt.QueuedConnection)
        self.inference_thread.start()
    
    def stop_inference(self):
        """추론 중지"""
        if self.inference_thread:
            self.inference_thread.stop()
    
    def on_progress(self, current, total, filename):
        """진행 상황 업데이트"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} - {filename}")
        self.result_count_label.setText(f"{current} / {total}")
    
    def on_preview(self, image_array, filepath, predicted_class, confidence):
        """미리보기 업데이트 (10장마다 호출) - 이미지 직접 표시"""
        # numpy array를 QPixmap으로 변환하여 직접 표시
        h, w, ch = image_array.shape
        qimg = QImage(image_array.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        
        # 이미지 뷰어에 직접 표시
        self.image_viewer._scene.clear()
        self.image_viewer._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.image_viewer._scene.addItem(self.image_viewer._pixmap_item)
        self.image_viewer._scene.setSceneRect(QRectF(pixmap.rect()))
        self.image_viewer.fit_in_view()
        
        # 파일명 표시
        filename = Path(filepath).name
        self.file_info_label.setText(f"미리보기: {filename}")
        
        # 예측 결과 표시
        class_colors = {"Normal": "#4caf50", "Twist": "#ff9800", "Hook": "#f44336"}
        color = class_colors.get(predicted_class, "#4a9eff")
        self.prediction_label.setText(f"예측: {predicted_class} ({confidence*100:.1f}%)")
        self.prediction_label.setStyleSheet(f"QLabel {{ color: {color}; padding: 10px; background-color: #2a2a2a; border-radius: 5px; font-size: 16pt; }}")
    
    def on_finished(self, elapsed_time, total_images, all_results):
        """추론 완료 - 결과 리스트 일괄 생성"""
        # 타이머 정지
        self.ui_update_timer.stop()
        
        # 결과 데이터 저장 및 리스트 일괄 생성
        self.result_list.setUpdatesEnabled(False)
        
        for filepath, predicted_class, confidence, all_probs in all_results:
            # 데이터 저장
            self.results[filepath] = (predicted_class, confidence, all_probs)
            
            # 리스트 아이템 생성
            filename = Path(filepath).name
            item_text = f"[{predicted_class}] {confidence*100:.1f}% - {filename}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, filepath)
            
            # 클래스별 색상
            if predicted_class == "Normal":
                item.setForeground(QColor("#4caf50"))
            elif predicted_class == "Twist":
                item.setForeground(QColor("#ff9800"))
            elif predicted_class == "Hook":
                item.setForeground(QColor("#f44336"))
            
            self.result_list.addItem(item)
        
        self.result_list.setUpdatesEnabled(True)
        
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # 통계 업데이트
        self.update_statistics()
        
        # 시간 정보 계산
        processed = len(self.results)
        if elapsed_time > 0:
            fps = processed / elapsed_time
            avg_time = elapsed_time / processed * 1000 if processed > 0 else 0  # ms per image
        else:
            fps = 0
            avg_time = 0
        
        # 시간 포맷팅
        if elapsed_time >= 60:
            time_str = f"{int(elapsed_time // 60)}분 {elapsed_time % 60:.1f}초"
        else:
            time_str = f"{elapsed_time:.2f}초"
        
        msg = (f"추론 완료!\n\n"
               f"📊 처리 결과\n"
               f"  • 총 이미지: {processed}개\n"
               f"  • 소요 시간: {time_str}\n"
               f"  • 처리 속도: {fps:.1f} FPS\n"
               f"  • 이미지당 평균: {avg_time:.1f} ms")
        
        QMessageBox.information(self, "완료", msg)
    
    def on_error(self, error_msg):
        """에러 처리"""
        # 타이머 정지
        self.ui_update_timer.stop()
        
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "오류", f"추론 중 오류 발생:\n{error_msg}")
    
    def filter_results(self, filter_text):
        """결과 필터링"""
        for i in range(self.result_list.count()):
            item = self.result_list.item(i)
            filepath = item.data(Qt.UserRole)
            if filepath in self.results:
                predicted_class = self.results[filepath][0]
                if filter_text == "전체" or predicted_class == filter_text:
                    item.setHidden(False)
                else:
                    item.setHidden(True)
    
    def on_result_selected(self, row):
        """결과 선택 시"""
        if row < 0:
            return
        
        item = self.result_list.item(row)
        if not item:
            return
        
        filepath = item.data(Qt.UserRole)
        if filepath not in self.results:
            return
        
        # 현재 선택된 이미지 경로 저장
        self.current_selected_path = filepath
        self.showing_heatmap = False
        
        predicted_class, confidence, all_probs = self.results[filepath]
        
        # 이미지 표시
        self.image_viewer.set_image(filepath)
        
        # 예측 결과 표시
        if predicted_class == "Normal":
            color = "#4caf50"
        elif predicted_class == "Twist":
            color = "#ff9800"
        else:
            color = "#f44336"
        
        self.prediction_label.setText(f"예측: {predicted_class} ({confidence*100:.1f}%)")
        self.prediction_label.setStyleSheet(f"QLabel {{ color: {color}; padding: 10px; background-color: #2a2a2a; border-radius: 5px; font-size: 16pt; }}")
        
        # 모든 확률 표시
        prob_text = " | ".join([f"{name}: {prob*100:.1f}%" for name, prob in all_probs])
        self.prob_label.setText(prob_text)
        
        # 파일 정보
        file_path = Path(filepath)
        file_size = file_path.stat().st_size / 1024
        self.file_info_label.setText(f"{file_path.name} ({file_size:.1f} KB)")
        
        # 히트맵 버튼 활성화
        self.heatmap_btn.setEnabled(True)
        self.show_original_btn.setEnabled(False)
        
        # 히트맵 캐시 확인
        if filepath in self.heatmap_cache:
            self.heatmap_status_label.setText("✓ 히트맵 캐시됨")
        else:
            self.heatmap_status_label.setText("")
    
    def generate_heatmap(self):
        """선택된 이미지의 히트맵 생성"""
        if not self.current_selected_path or not self.model_path:
            return
        
        # 캐시 확인
        if self.current_selected_path in self.heatmap_cache:
            self._display_heatmap(self.heatmap_cache[self.current_selected_path])
            return
        
        # 버튼 비활성화 및 상태 표시
        self.heatmap_btn.setEnabled(False)
        self.heatmap_status_label.setText("🔄 히트맵 생성 중...")
        
        # 히트맵 생성 스레드 시작
        self.heatmap_thread = HeatmapThread(
            str(self.model_path),
            self.current_selected_path,
            use_gpu=True
        )
        self.heatmap_thread.finished.connect(self.on_heatmap_generated)
        self.heatmap_thread.error.connect(self.on_heatmap_error)
        self.heatmap_thread.start()
    
    def on_heatmap_generated(self, heatmap_array, filepath):
        """히트맵 생성 완료"""
        # numpy array를 QPixmap으로 변환
        h, w, ch = heatmap_array.shape
        qimg = QImage(heatmap_array.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        
        # 캐시에 저장
        self.heatmap_cache[filepath] = pixmap
        
        # 현재 선택된 이미지와 일치하면 표시
        if filepath == self.current_selected_path:
            self._display_heatmap(pixmap)
        
        self.heatmap_btn.setEnabled(True)
        self.heatmap_status_label.setText("✓ 히트맵 생성 완료")
    
    def on_heatmap_error(self, error_msg):
        """히트맵 생성 오류"""
        self.heatmap_btn.setEnabled(True)
        self.heatmap_status_label.setText(f"❌ 오류")
        QMessageBox.warning(self, "히트맵 오류", f"히트맵 생성 실패:\n{error_msg}")
    
    def _display_heatmap(self, pixmap):
        """히트맵 표시"""
        self.image_viewer._scene.clear()
        self.image_viewer._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.image_viewer._scene.addItem(self.image_viewer._pixmap_item)
        self.image_viewer._scene.setSceneRect(QRectF(pixmap.rect()))
        self.image_viewer.fit_in_view()
        
        self.showing_heatmap = True
        self.show_original_btn.setEnabled(True)
    
    def show_original_image(self):
        """원본 이미지 표시"""
        if self.current_selected_path:
            self.image_viewer.set_image(self.current_selected_path)
            self.showing_heatmap = False
            self.show_original_btn.setEnabled(False)
    
    def update_statistics(self):
        """통계 업데이트"""
        if not self.results:
            return
        
        # 클래스별 집계
        class_counts = defaultdict(list)
        all_confidences = []
        
        for filepath, (predicted_class, confidence, _) in self.results.items():
            class_counts[predicted_class].append(confidence)
            all_confidences.append(confidence)
        
        total = len(self.results)
        
        # 요약 레이블 업데이트
        self.total_label.setText(f"전체: {total}")
        
        normal_count = len(class_counts.get("Normal", []))
        twist_count = len(class_counts.get("Twist", []))
        hook_count = len(class_counts.get("Hook", []))
        
        self.normal_count_label.setText(f"Normal: {normal_count} ({normal_count/total*100:.1f}%)")
        self.twist_count_label.setText(f"Twist: {twist_count} ({twist_count/total*100:.1f}%)")
        self.hook_count_label.setText(f"Hook: {hook_count} ({hook_count/total*100:.1f}%)")
        
        # 신뢰도 통계
        if all_confidences:
            self.avg_conf_label.setText(f"평균 신뢰도: {np.mean(all_confidences)*100:.1f}%")
            self.min_conf_label.setText(f"최소 신뢰도: {np.min(all_confidences)*100:.1f}%")
            self.max_conf_label.setText(f"최대 신뢰도: {np.max(all_confidences)*100:.1f}%")
        
        # 테이블 업데이트
        self.stats_table.setRowCount(3)
        for i, class_name in enumerate(["Normal", "Twist", "Hook"]):
            confs = class_counts.get(class_name, [])
            count = len(confs)
            ratio = count / total * 100 if total > 0 else 0
            avg_conf = np.mean(confs) * 100 if confs else 0
            min_conf = np.min(confs) * 100 if confs else 0
            
            self.stats_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{ratio:.1f}"))
            self.stats_table.setItem(i, 3, QTableWidgetItem(f"{avg_conf:.1f}%"))
            self.stats_table.setItem(i, 4, QTableWidgetItem(f"{min_conf:.1f}%"))
        
        # 낮은 신뢰도 이미지 목록
        self.low_conf_list.clear()
        for filepath, (predicted_class, confidence, _) in self.results.items():
            if confidence < 0.8:
                filename = Path(filepath).name
                item_text = f"[{predicted_class}] {confidence*100:.1f}% - {filename}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, filepath)
                self.low_conf_list.addItem(item)
    
    def on_low_conf_item_clicked(self, item):
        """낮은 신뢰도 이미지 클릭 시"""
        filepath = item.data(Qt.UserRole)
        # 메인 리스트에서 해당 항목 찾기
        for i in range(self.result_list.count()):
            list_item = self.result_list.item(i)
            if list_item.data(Qt.UserRole) == filepath:
                self.result_list.setCurrentRow(i)
                break
    
    def select_previous(self):
        """이전 항목 선택"""
        current = self.result_list.currentRow()
        if current > 0:
            self.result_list.setCurrentRow(current - 1)
    
    def select_next(self):
        """다음 항목 선택"""
        current = self.result_list.currentRow()
        if current < self.result_list.count() - 1:
            self.result_list.setCurrentRow(current + 1)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ModelTestGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
