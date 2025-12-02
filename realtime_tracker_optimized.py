"""
권취 줄 실시간 추적 시스템 (최적화 버전)
- 추론과 UI 분리 (멀티스레드)
- 배치 처리로 효율성 향상
- UI 업데이트 주기 조절로 버벅임 최소화
- 그래프는 일정 간격으로만 업데이트
"""
import sys
import cv2
import numpy as np
import pandas as pd
import nrt
from pathlib import Path
from datetime import datetime
import json
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QSpinBox,
    QCheckBox, QProgressBar, QStatusBar, QMessageBox, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 기본 경로 설정
DEFAULT_MODEL_PATH = r"D:\LLM_Dataset\models\Trace_Coil.net"
DEFAULT_IMAGE_DIR = r"K:\LLM Image_Storage\A line-2025-07-25_09-49-08(정상)"
DEFAULT_OUTPUT_DIR = r"D:\LLM_Dataset\tracking_results"
INPUT_SIZE = 512


def imread_korean(path):
    """한글 경로 이미지 로드"""
    stream = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


class InferenceWorker(QThread):
    """추론 전용 워커 스레드 - UI와 완전 분리"""
    result_ready = pyqtSignal(dict)  # 개별 결과
    batch_complete = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, model_path, image_files, input_size=512):
        super().__init__()
        self.model_path = model_path
        self.image_files = image_files
        self.input_size = input_size
        self.is_running = True
        self.is_paused = False
        self.mutex = QMutex()
    
    def run(self):
        try:
            # 모델 로드
            predictor = nrt.Predictor(self.model_path)
            
            for idx, img_path in enumerate(self.image_files):
                # 중지/일시정지 체크
                with QMutexLocker(self.mutex):
                    if not self.is_running:
                        break
                
                while self.is_paused and self.is_running:
                    self.msleep(50)
                
                # 이미지 로드 및 추론
                result = self.process_single_image(predictor, idx, img_path)
                
                # 결과 전송 (UI 스레드로)
                self.result_ready.emit(result)
                
                # 진행률 (50프레임마다)
                if (idx + 1) % 50 == 0:
                    self.batch_complete.emit(idx + 1, len(self.image_files))
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
    
    def process_single_image(self, predictor, idx, img_path):
        """단일 이미지 처리"""
        result = {
            'frame': idx,
            'file': img_path.name,
            'detected': False,
            'image_path': str(img_path)
        }
        
        img = imread_korean(img_path)
        if img is None:
            return result
        
        orig_h, orig_w = img.shape[:2]
        result['orig_size'] = (orig_w, orig_h)
        
        # 전처리 및 추론
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        input_data = nrt.Input()
        image_buff = nrt.NDBuffer.from_numpy(img_resized)
        input_data.extend(image_buff)
        
        pred_result = predictor.predict(input_data)
        input_data.clear()
        
        # 결과 파싱
        if hasattr(pred_result, 'bboxes') and pred_result.bboxes.get_count() > 0:
            best_bbox = None
            best_area = 0
            for i in range(pred_result.bboxes.get_count()):
                bbox = pred_result.bboxes.get(i)
                area = bbox.rect.width * bbox.rect.height
                if area > best_area:
                    best_area = area
                    best_bbox = bbox
            
            if best_bbox:
                scale_x = orig_w / self.input_size
                scale_y = orig_h / self.input_size
                
                result.update({
                    'detected': True,
                    'center_x': int((best_bbox.rect.x + best_bbox.rect.width / 2) * scale_x),
                    'center_y': int((best_bbox.rect.y + best_bbox.rect.height / 2) * scale_y),
                    'bbox_x': int(best_bbox.rect.x * scale_x),
                    'bbox_y': int(best_bbox.rect.y * scale_y),
                    'bbox_w': int(best_bbox.rect.width * scale_x),
                    'bbox_h': int(best_bbox.rect.height * scale_y),
                    'class_idx': best_bbox.class_idx
                })
        
        return result
    
    def pause(self):
        with QMutexLocker(self.mutex):
            self.is_paused = True
    
    def resume(self):
        with QMutexLocker(self.mutex):
            self.is_paused = False
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.is_running = False
            self.is_paused = False


class LazyGraphCanvas(FigureCanvas):
    """지연 업데이트 그래프 - 버퍼링 후 일괄 업데이트"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        
        self.x_data = []
        self.y_data = []
        self.frames = []
        
        self.init_plots()
    
    def init_plots(self):
        """그래프 초기화"""
        self.ax1.set_title('X 좌표 변화')
        self.ax1.set_xlabel('프레임')
        self.ax1.set_ylabel('X (픽셀)')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Y 좌표 변화')
        self.ax2.set_xlabel('프레임')
        self.ax2.set_ylabel('Y (픽셀)')
        self.ax2.grid(True, alpha=0.3)
        
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=1)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=1)
        
        self.fig.tight_layout()
    
    def add_data(self, frame, center_x, center_y):
        """데이터 추가 (그리기는 하지 않음)"""
        self.frames.append(frame)
        self.x_data.append(center_x)
        self.y_data.append(center_y)
    
    def refresh_plot(self):
        """그래프 갱신 (호출 시에만 실제 그리기)"""
        if not self.frames:
            return
        
        try:
            self.line1.set_data(self.frames, self.x_data)
            self.line2.set_data(self.frames, self.y_data)
            
            # 축 범위 조정
            self.ax1.set_xlim(0, max(self.frames) + 10)
            self.ax2.set_xlim(0, max(self.frames) + 10)
            
            if self.x_data:
                margin_x = (max(self.x_data) - min(self.x_data)) * 0.1 + 10
                self.ax1.set_ylim(min(self.x_data) - margin_x, max(self.x_data) + margin_x)
            
            if self.y_data:
                margin_y = (max(self.y_data) - min(self.y_data)) * 0.1 + 10
                self.ax2.set_ylim(min(self.y_data) - margin_y, max(self.y_data) + margin_y)
            
            self.draw_idle()  # draw() 대신 draw_idle() 사용 - 더 부드러움
        except Exception as e:
            print(f"Graph update error: {e}")
    
    def clear_data(self):
        """데이터 초기화"""
        self.frames = []
        self.x_data = []
        self.y_data = []
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.draw_idle()
    
    def save_plot(self, filepath):
        """그래프 저장"""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')


class OptimizedTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("권취 줄 추적 시스템 (최적화)")
        self.setGeometry(50, 50, 1600, 950)
        
        # 데이터
        self.model_path = DEFAULT_MODEL_PATH
        self.image_dir = None
        self.image_files = []
        self.tracking_results = []
        self.trajectory = deque(maxlen=100)  # 최근 100개만 유지
        
        # 결과 버퍼 (UI 업데이트용)
        self.result_buffer = deque(maxlen=10)
        self.latest_result = None
        
        # 워커 스레드
        self.inference_worker = None
        
        # 표시 옵션
        self.show_bbox = True
        self.show_center = True
        self.show_trajectory = True
        
        # UI 업데이트 타이머 (추론과 분리)
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_display)
        
        # 그래프 업데이트 타이머 (더 낮은 빈도)
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graph)
        
        # 통계
        self.processed_count = 0
        self.detected_count = 0
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 왼쪽: 이미지 뷰어
        left_panel = QVBoxLayout()
        
        # 이미지 표시
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 2px solid #444;")
        left_panel.addWidget(self.image_label)
        
        # 진행률
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("0 / 0")
        self.progress_label.setMinimumWidth(120)
        progress_layout.addWidget(self.progress_label)
        left_panel.addLayout(progress_layout)
        
        # 컨트롤 버튼
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ 추론 시작")
        self.start_btn.clicked.connect(self.start_inference)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        control_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ 일시정지")
        self.pause_btn.clicked.connect(self.pause_inference)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ 중지")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        control_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("💾 결과 저장")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        left_panel.addLayout(control_layout)
        
        main_layout.addLayout(left_panel, stretch=2)
        
        # 오른쪽: 정보 + 그래프
        right_panel = QVBoxLayout()
        
        # 탭 위젯
        self.tab_widget = QTabWidget()
        
        # 탭 1: 설정 및 정보
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # 모델 설정
        model_group = QGroupBox("모델 설정")
        model_layout = QVBoxLayout(model_group)
        
        self.model_label = QLabel(f"모델: {Path(self.model_path).name}")
        self.model_label.setWordWrap(True)
        model_layout.addWidget(self.model_label)
        
        self.load_model_btn = QPushButton("모델 파일 변경")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        settings_layout.addWidget(model_group)
        
        # 이미지 폴더 설정
        folder_group = QGroupBox("이미지 폴더")
        folder_layout = QVBoxLayout(folder_group)
        
        self.folder_label = QLabel("폴더: 선택되지 않음")
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(self.folder_label)
        
        self.load_folder_btn = QPushButton("이미지 폴더 선택")
        self.load_folder_btn.clicked.connect(self.load_folder)
        folder_layout.addWidget(self.load_folder_btn)
        
        settings_layout.addWidget(folder_group)
        
        # 성능 설정
        perf_group = QGroupBox("성능 설정")
        perf_layout = QVBoxLayout(perf_group)
        
        ui_rate_layout = QHBoxLayout()
        ui_rate_layout.addWidget(QLabel("UI 갱신 주기:"))
        self.ui_rate_spin = QSpinBox()
        self.ui_rate_spin.setRange(50, 500)
        self.ui_rate_spin.setValue(100)
        self.ui_rate_spin.setSuffix(" ms")
        ui_rate_layout.addWidget(self.ui_rate_spin)
        perf_layout.addLayout(ui_rate_layout)
        
        graph_rate_layout = QHBoxLayout()
        graph_rate_layout.addWidget(QLabel("그래프 갱신 주기:"))
        self.graph_rate_spin = QSpinBox()
        self.graph_rate_spin.setRange(500, 5000)
        self.graph_rate_spin.setValue(1000)
        self.graph_rate_spin.setSuffix(" ms")
        graph_rate_layout.addWidget(self.graph_rate_spin)
        perf_layout.addLayout(graph_rate_layout)
        
        settings_layout.addWidget(perf_group)
        
        # 표시 옵션
        display_group = QGroupBox("표시 옵션")
        display_layout = QVBoxLayout(display_group)
        
        self.bbox_check = QCheckBox("Bounding Box 표시")
        self.bbox_check.setChecked(True)
        self.bbox_check.stateChanged.connect(lambda: setattr(self, 'show_bbox', self.bbox_check.isChecked()))
        display_layout.addWidget(self.bbox_check)
        
        self.center_check = QCheckBox("중심점 표시")
        self.center_check.setChecked(True)
        self.center_check.stateChanged.connect(lambda: setattr(self, 'show_center', self.center_check.isChecked()))
        display_layout.addWidget(self.center_check)
        
        self.traj_check = QCheckBox("궤적 표시")
        self.traj_check.setChecked(True)
        self.traj_check.stateChanged.connect(lambda: setattr(self, 'show_trajectory', self.traj_check.isChecked()))
        display_layout.addWidget(self.traj_check)
        
        settings_layout.addWidget(display_group)
        
        # 현재 프레임 정보
        info_group = QGroupBox("현재 프레임 정보")
        info_layout = QVBoxLayout(info_group)
        
        self.info_frame = QLabel("프레임: -")
        self.info_frame.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_frame)
        
        self.info_detection = QLabel("탐지: -")
        self.info_detection.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_detection)
        
        self.info_center = QLabel("중심점: -")
        self.info_center.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_center)
        
        self.info_bbox = QLabel("BBox: -")
        self.info_bbox.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_bbox)
        
        settings_layout.addWidget(info_group)
        
        # 실시간 통계
        stats_group = QGroupBox("실시간 통계")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("추론을 시작하세요")
        self.stats_label.setFont(QFont("Consolas", 9))
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        settings_layout.addWidget(stats_group)
        settings_layout.addStretch()
        
        self.tab_widget.addTab(settings_tab, "설정 / 정보")
        
        # 탭 2: 그래프
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        
        self.graph_canvas = LazyGraphCanvas(self)
        graph_layout.addWidget(self.graph_canvas)
        
        graph_btn_layout = QHBoxLayout()
        self.clear_graph_btn = QPushButton("그래프 초기화")
        self.clear_graph_btn.clicked.connect(self.graph_canvas.clear_data)
        graph_btn_layout.addWidget(self.clear_graph_btn)
        
        self.save_graph_btn = QPushButton("그래프 저장")
        self.save_graph_btn.clicked.connect(self.save_graph)
        graph_btn_layout.addWidget(self.save_graph_btn)
        
        self.refresh_graph_btn = QPushButton("그래프 새로고침")
        self.refresh_graph_btn.clicked.connect(self.graph_canvas.refresh_plot)
        graph_btn_layout.addWidget(self.refresh_graph_btn)
        
        graph_layout.addLayout(graph_btn_layout)
        
        self.tab_widget.addTab(graph_tab, "실시간 그래프")
        
        right_panel.addWidget(self.tab_widget)
        
        main_layout.addLayout(right_panel, stretch=1)
        
        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("준비 - 이미지 폴더를 선택하세요")
        
        # 기본 폴더 로드
        self.load_default_folder()
    
    def load_default_folder(self):
        """기본 폴더 로드"""
        folder = Path(DEFAULT_IMAGE_DIR)
        if folder.exists():
            self.load_folder_path(folder)
    
    def load_model(self):
        """모델 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "모델 파일 선택", str(Path(self.model_path).parent), "NET Files (*.net)"
        )
        if file_path:
            self.model_path = file_path
            self.model_label.setText(f"모델: {Path(file_path).name}")
    
    def load_folder(self):
        """이미지 폴더 선택"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "이미지 폴더 선택", DEFAULT_IMAGE_DIR
        )
        if folder_path:
            self.load_folder_path(Path(folder_path))
    
    def load_folder_path(self, folder_path):
        """폴더 로드"""
        self.image_dir = folder_path
        self.image_files = sorted([f for f in folder_path.glob("*.jpg")])
        
        if self.image_files:
            self.folder_label.setText(f"폴더: {folder_path.name}\n({len(self.image_files)}장)")
            self.progress_bar.setMaximum(len(self.image_files))
            self.statusBar.showMessage(f"이미지 로드 완료: {len(self.image_files)}장")
            self.start_btn.setEnabled(True)
    
    def start_inference(self):
        """추론 시작"""
        if not self.image_files:
            QMessageBox.warning(self, "경고", "이미지 폴더를 먼저 선택하세요.")
            return
        
        # 초기화
        self.tracking_results = []
        self.trajectory.clear()
        self.result_buffer.clear()
        self.latest_result = None
        self.processed_count = 0
        self.detected_count = 0
        self.graph_canvas.clear_data()
        
        # 워커 스레드 시작
        self.inference_worker = InferenceWorker(self.model_path, self.image_files, INPUT_SIZE)
        self.inference_worker.result_ready.connect(self.on_result_ready)
        self.inference_worker.batch_complete.connect(self.on_batch_complete)
        self.inference_worker.finished.connect(self.on_inference_finished)
        self.inference_worker.error.connect(self.on_inference_error)
        self.inference_worker.start()
        
        # UI 업데이트 타이머 시작 (추론과 분리)
        self.ui_timer.start(self.ui_rate_spin.value())
        self.graph_timer.start(self.graph_rate_spin.value())
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.load_folder_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        
        self.statusBar.showMessage("추론 진행 중... (백그라운드)")
    
    def on_result_ready(self, result):
        """추론 결과 수신 (빠르게 버퍼에만 저장)"""
        self.processed_count += 1
        self.latest_result = result
        
        if result['detected']:
            self.detected_count += 1
            self.trajectory.append((result['center_x'], result['center_y']))
            
            # 그래프 데이터 추가 (그리기는 하지 않음)
            self.graph_canvas.add_data(result['frame'], result['center_x'], result['center_y'])
            
            # 결과 저장
            self.tracking_results.append({
                'frame': result['frame'],
                'filename': result['file'],
                'center_x': result['center_x'],
                'center_y': result['center_y'],
                'bbox_x': result['bbox_x'],
                'bbox_y': result['bbox_y'],
                'bbox_w': result['bbox_w'],
                'bbox_h': result['bbox_h'],
                'class_idx': result['class_idx']
            })
    
    def on_batch_complete(self, current, total):
        """배치 완료 시 진행률 업데이트"""
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current} / {total}")
    
    def update_ui_display(self):
        """UI 디스플레이 업데이트 (타이머에 의해 호출)"""
        if self.latest_result is None:
            return
        
        result = self.latest_result
        
        # 진행률 업데이트
        self.progress_bar.setValue(self.processed_count)
        self.progress_label.setText(f"{self.processed_count} / {len(self.image_files)}")
        
        # 이미지 로드 및 표시
        try:
            img = imread_korean(result['image_path'])
            if img is not None:
                self.display_result(img, result)
        except Exception as e:
            pass
        
        # 정보 업데이트
        self.info_frame.setText(f"프레임: {result['frame']}")
        
        if result['detected']:
            self.info_detection.setText("탐지: ✓ 감지됨")
            self.info_detection.setStyleSheet("color: green; font-weight: bold;")
            self.info_center.setText(f"중심점: ({result['center_x']}, {result['center_y']})")
            self.info_bbox.setText(f"BBox: ({result['bbox_x']}, {result['bbox_y']}, {result['bbox_w']}, {result['bbox_h']})")
        else:
            self.info_detection.setText("탐지: ✗ 미감지")
            self.info_detection.setStyleSheet("color: red;")
            self.info_center.setText("중심점: -")
            self.info_bbox.setText("BBox: -")
        
        # 통계 업데이트
        detection_rate = self.detected_count / self.processed_count * 100 if self.processed_count > 0 else 0
        self.stats_label.setText(
            f"처리: {self.processed_count} / {len(self.image_files)}\n"
            f"탐지: {self.detected_count}개\n"
            f"탐지율: {detection_rate:.1f}%\n"
            f"진행률: {self.processed_count / len(self.image_files) * 100:.1f}%"
        )
    
    def update_graph(self):
        """그래프 업데이트 (낮은 빈도)"""
        self.graph_canvas.refresh_plot()
    
    def display_result(self, img, result):
        """결과 이미지 표시"""
        img_display = img.copy()
        
        # 궤적 그리기
        if self.show_trajectory and len(self.trajectory) > 1:
            traj_list = list(self.trajectory)
            for i in range(1, len(traj_list)):
                alpha = i / len(traj_list)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(img_display, traj_list[i-1], traj_list[i], color, 2)
        
        # 탐지 결과 그리기
        if result['detected']:
            cx, cy = result['center_x'], result['center_y']
            bx, by = result['bbox_x'], result['bbox_y']
            bw, bh = result['bbox_w'], result['bbox_h']
            
            if self.show_bbox:
                cv2.rectangle(img_display, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            
            if self.show_center:
                cv2.circle(img_display, (cx, cy), 10, (0, 0, 255), -1)
                cv2.circle(img_display, (cx, cy), 15, (255, 255, 255), 2)
                cv2.putText(img_display, f"({cx}, {cy})", (cx + 20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 프레임 정보
        cv2.putText(img_display, f"Frame: {result['frame']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 이미지 변환
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        
        # 크기 조절
        label_w = self.image_label.width() - 10
        label_h = self.image_label.height() - 10
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        qimg = QImage(img_resized.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
    
    def pause_inference(self):
        """추론 일시정지/재개"""
        if self.inference_worker:
            if self.inference_worker.is_paused:
                self.inference_worker.resume()
                self.pause_btn.setText("⏸ 일시정지")
                self.statusBar.showMessage("추론 재개...")
            else:
                self.inference_worker.pause()
                self.pause_btn.setText("▶ 계속")
                self.statusBar.showMessage("추론 일시정지")
    
    def stop_inference(self):
        """추론 중지"""
        if self.inference_worker:
            self.inference_worker.stop()
            self.inference_worker.wait()
        self.on_inference_finished()
    
    def on_inference_finished(self):
        """추론 완료"""
        # 타이머 중지
        self.ui_timer.stop()
        self.graph_timer.stop()
        
        # 마지막 그래프 업데이트
        self.graph_canvas.refresh_plot()
        
        # UI 상태 복원
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.load_folder_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 일시정지")
        
        self.statusBar.showMessage(f"추론 완료! 총 {len(self.tracking_results)}개 탐지")
        
        if self.processed_count > 0:
            QMessageBox.information(self, "완료", 
                f"추론이 완료되었습니다.\n\n"
                f"총 프레임: {self.processed_count}\n"
                f"탐지된 프레임: {len(self.tracking_results)}\n"
                f"탐지율: {len(self.tracking_results)/self.processed_count*100:.1f}%"
            )
    
    def on_inference_error(self, error_msg):
        """추론 에러"""
        self.ui_timer.stop()
        self.graph_timer.stop()
        QMessageBox.critical(self, "오류", f"추론 중 오류 발생:\n{error_msg}")
        self.on_inference_finished()
    
    def save_results(self):
        """결과 저장"""
        if not self.tracking_results:
            QMessageBox.warning(self, "경고", "저장할 결과가 없습니다.")
            return
        
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV 저장
        df = pd.DataFrame(self.tracking_results)
        csv_path = output_dir / f"optimized_tracking_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # JSON 저장
        json_path = output_dir / f"optimized_tracking_{timestamp}.json"
        json_data = {
            'metadata': {
                'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source_dir': str(self.image_dir),
                'model_path': self.model_path,
                'total_frames': len(self.image_files),
                'detected_frames': len(self.tracking_results)
            },
            'tracking_data': self.tracking_results
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # 그래프 저장
        plot_path = output_dir / f"optimized_plot_{timestamp}.png"
        self.graph_canvas.save_plot(plot_path)
        
        self.statusBar.showMessage(f"결과 저장 완료: {output_dir}")
        
        QMessageBox.information(self, "저장 완료", 
            f"결과가 저장되었습니다.\n\n"
            f"CSV: {csv_path.name}\n"
            f"JSON: {json_path.name}\n"
            f"그래프: {plot_path.name}"
        )
    
    def save_graph(self):
        """그래프만 저장"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "그래프 저장", DEFAULT_OUTPUT_DIR, "PNG Files (*.png)"
        )
        if file_path:
            self.graph_canvas.save_plot(file_path)
            self.statusBar.showMessage(f"그래프 저장: {file_path}")
    
    def closeEvent(self, event):
        """창 닫기"""
        if self.inference_worker and self.inference_worker.isRunning():
            reply = QMessageBox.question(self, '확인', 
                "추론이 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.inference_worker.stop()
                self.inference_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    tracker = OptimizedTracker()
    tracker.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
