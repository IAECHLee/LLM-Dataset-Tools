"""
권취 줄 실시간 추적 시스템
- NRT 모델을 이용한 실시간 Object Detection
- 추적 결과 시각화 및 좌표 저장
- 결과 그래프 생성
"""
import sys
import cv2
import numpy as np
import pandas as pd
import nrt
from pathlib import Path
from datetime import datetime
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QSpinBox,
    QCheckBox, QProgressBar, QStatusBar, QMessageBox, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
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


class InferenceThread(QThread):
    """추론 스레드"""
    progress = pyqtSignal(int, int, dict)  # current, total, result
    finished = pyqtSignal(list)  # all results
    error = pyqtSignal(str)
    
    def __init__(self, model_path, image_files, input_size=512):
        super().__init__()
        self.model_path = model_path
        self.image_files = image_files
        self.input_size = input_size
        self.is_running = True
        self.is_paused = False
    
    def run(self):
        try:
            # 모델 로드
            predictor = nrt.Predictor(self.model_path)
            results = []
            
            for idx, img_path in enumerate(self.image_files):
                if not self.is_running:
                    break
                
                while self.is_paused and self.is_running:
                    self.msleep(100)
                
                # 이미지 로드
                img = imread_korean(img_path)
                if img is None:
                    self.progress.emit(idx, len(self.image_files), {
                        'frame': idx,
                        'file': img_path.name,
                        'detected': False,
                        'image': None
                    })
                    continue
                
                orig_h, orig_w = img.shape[:2]
                
                # 전처리
                img_resized = cv2.resize(img, (self.input_size, self.input_size))
                
                # 추론
                input_data = nrt.Input()
                image_buff = nrt.NDBuffer.from_numpy(img_resized)
                input_data.extend(image_buff)
                
                result = predictor.predict(input_data)
                input_data.clear()
                
                # 결과 파싱
                detection_result = {
                    'frame': idx,
                    'file': img_path.name,
                    'detected': False,
                    'image': img,
                    'orig_size': (orig_w, orig_h)
                }
                
                if hasattr(result, 'bboxes') and result.bboxes.get_count() > 0:
                    # 가장 큰 bbox 선택
                    best_bbox = None
                    best_area = 0
                    for i in range(result.bboxes.get_count()):
                        bbox = result.bboxes.get(i)
                        area = bbox.rect.width * bbox.rect.height
                        if area > best_area:
                            best_area = area
                            best_bbox = bbox
                    
                    if best_bbox:
                        scale_x = orig_w / self.input_size
                        scale_y = orig_h / self.input_size
                        
                        orig_x = int(best_bbox.rect.x * scale_x)
                        orig_y = int(best_bbox.rect.y * scale_y)
                        orig_w_box = int(best_bbox.rect.width * scale_x)
                        orig_h_box = int(best_bbox.rect.height * scale_y)
                        center_x = orig_x + orig_w_box // 2
                        center_y = orig_y + orig_h_box // 2
                        
                        detection_result.update({
                            'detected': True,
                            'center_x': center_x,
                            'center_y': center_y,
                            'bbox_x': orig_x,
                            'bbox_y': orig_y,
                            'bbox_w': orig_w_box,
                            'bbox_h': orig_h_box,
                            'class_idx': best_bbox.class_idx
                        })
                        
                        results.append({
                            'frame': idx,
                            'filename': img_path.name,
                            'center_x': center_x,
                            'center_y': center_y,
                            'bbox_x': orig_x,
                            'bbox_y': orig_y,
                            'bbox_w': orig_w_box,
                            'bbox_h': orig_h_box,
                            'class_idx': best_bbox.class_idx
                        })
                
                self.progress.emit(idx, len(self.image_files), detection_result)
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False
    
    def stop(self):
        self.is_running = False
        self.is_paused = False


class GraphCanvas(FigureCanvas):
    """실시간 그래프 캔버스"""
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
    
    def update_data(self, frame, center_x, center_y):
        """데이터 추가 및 그래프 업데이트"""
        self.frames.append(frame)
        self.x_data.append(center_x)
        self.y_data.append(center_y)
        
        # 데이터 업데이트
        self.line1.set_data(self.frames, self.x_data)
        self.line2.set_data(self.frames, self.y_data)
        
        # 축 범위 조정
        if self.frames:
            self.ax1.set_xlim(0, max(self.frames) + 10)
            self.ax2.set_xlim(0, max(self.frames) + 10)
            
            if self.x_data:
                margin_x = (max(self.x_data) - min(self.x_data)) * 0.1 + 10
                self.ax1.set_ylim(min(self.x_data) - margin_x, max(self.x_data) + margin_x)
            
            if self.y_data:
                margin_y = (max(self.y_data) - min(self.y_data)) * 0.1 + 10
                self.ax2.set_ylim(min(self.y_data) - margin_y, max(self.y_data) + margin_y)
        
        self.draw()
    
    def clear_data(self):
        """데이터 초기화"""
        self.frames = []
        self.x_data = []
        self.y_data = []
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.draw()
    
    def save_plot(self, filepath):
        """그래프 저장"""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')


class RealtimeTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("권취 줄 실시간 추적 시스템")
        self.setGeometry(50, 50, 1600, 950)
        
        # 데이터
        self.model_path = DEFAULT_MODEL_PATH
        self.image_dir = None
        self.image_files = []
        self.tracking_results = []
        self.trajectory = []  # 궤적 데이터
        
        # 스레드
        self.inference_thread = None
        
        # 표시 옵션
        self.show_bbox = True
        self.show_center = True
        self.show_trajectory = True
        
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
        self.progress_label.setMinimumWidth(100)
        progress_layout.addWidget(self.progress_label)
        left_panel.addLayout(progress_layout)
        
        # 컨트롤 버튼
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ 추론 시작")
        self.start_btn.clicked.connect(self.start_inference)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        control_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ 일시정지")
        self.pause_btn.clicked.connect(self.pause_inference)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ 중지")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
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
        
        # 통계
        stats_group = QGroupBox("실시간 통계")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("추론을 시작하세요")
        self.stats_label.setFont(QFont("Consolas", 9))
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        settings_layout.addWidget(stats_group)
        settings_layout.addStretch()
        
        self.tab_widget.addTab(settings_tab, "설정 / 정보")
        
        # 탭 2: 실시간 그래프
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        
        self.graph_canvas = GraphCanvas(self)
        graph_layout.addWidget(self.graph_canvas)
        
        graph_btn_layout = QHBoxLayout()
        self.clear_graph_btn = QPushButton("그래프 초기화")
        self.clear_graph_btn.clicked.connect(self.graph_canvas.clear_data)
        graph_btn_layout.addWidget(self.clear_graph_btn)
        
        self.save_graph_btn = QPushButton("그래프 저장")
        self.save_graph_btn.clicked.connect(self.save_graph)
        graph_btn_layout.addWidget(self.save_graph_btn)
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
            self.statusBar.showMessage(f"모델 로드: {Path(file_path).name}")
    
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
        else:
            self.statusBar.showMessage("이미지를 찾을 수 없습니다")
    
    def start_inference(self):
        """추론 시작"""
        if not self.image_files:
            QMessageBox.warning(self, "경고", "이미지 폴더를 먼저 선택하세요.")
            return
        
        if not Path(self.model_path).exists():
            QMessageBox.warning(self, "경고", "모델 파일을 찾을 수 없습니다.")
            return
        
        # 초기화
        self.tracking_results = []
        self.trajectory = []
        self.graph_canvas.clear_data()
        
        # 스레드 시작
        self.inference_thread = InferenceThread(self.model_path, self.image_files, INPUT_SIZE)
        self.inference_thread.progress.connect(self.on_inference_progress)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.start()
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.load_folder_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        
        self.statusBar.showMessage("추론 진행 중...")
    
    def pause_inference(self):
        """추론 일시정지/재개"""
        if self.inference_thread:
            if self.inference_thread.is_paused:
                self.inference_thread.resume()
                self.pause_btn.setText("⏸ 일시정지")
                self.statusBar.showMessage("추론 재개...")
            else:
                self.inference_thread.pause()
                self.pause_btn.setText("▶ 계속")
                self.statusBar.showMessage("추론 일시정지")
    
    def stop_inference(self):
        """추론 중지"""
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread.wait()
            self.on_inference_finished(self.tracking_results)
    
    def on_inference_progress(self, current, total, result):
        """추론 진행 콜백"""
        # 진행률 업데이트
        self.progress_bar.setValue(current + 1)
        self.progress_label.setText(f"{current + 1} / {total}")
        
        # 이미지 표시
        if result.get('image') is not None:
            self.display_result(result)
        
        # 정보 업데이트
        self.info_frame.setText(f"프레임: {result['frame']}")
        
        if result['detected']:
            self.info_detection.setText("탐지: ✓ 감지됨")
            self.info_detection.setStyleSheet("color: green; font-weight: bold;")
            self.info_center.setText(f"중심점: ({result['center_x']}, {result['center_y']})")
            self.info_bbox.setText(f"BBox: ({result['bbox_x']}, {result['bbox_y']}, {result['bbox_w']}, {result['bbox_h']})")
            
            # 궤적 추가
            self.trajectory.append((result['center_x'], result['center_y']))
            if len(self.trajectory) > 100:
                self.trajectory.pop(0)
            
            # 그래프 업데이트
            self.graph_canvas.update_data(result['frame'], result['center_x'], result['center_y'])
            
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
        else:
            self.info_detection.setText("탐지: ✗ 미감지")
            self.info_detection.setStyleSheet("color: red;")
            self.info_center.setText("중심점: -")
            self.info_bbox.setText("BBox: -")
        
        # 통계 업데이트
        detection_rate = len(self.tracking_results) / (current + 1) * 100 if current >= 0 else 0
        self.stats_label.setText(
            f"처리: {current + 1} / {total}\n"
            f"탐지: {len(self.tracking_results)}개\n"
            f"탐지율: {detection_rate:.1f}%"
        )
    
    def display_result(self, result):
        """결과 이미지 표시"""
        img = result['image'].copy()
        
        # 궤적 그리기
        if self.show_trajectory and len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                alpha = i / len(self.trajectory)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(img, self.trajectory[i-1], self.trajectory[i], color, 2)
        
        # 탐지 결과 그리기
        if result['detected']:
            cx, cy = result['center_x'], result['center_y']
            bx, by = result['bbox_x'], result['bbox_y']
            bw, bh = result['bbox_w'], result['bbox_h']
            
            if self.show_bbox:
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            
            if self.show_center:
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
                cv2.circle(img, (cx, cy), 15, (255, 255, 255), 2)
                cv2.putText(img, f"({cx}, {cy})", (cx + 20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 프레임 정보
        cv2.putText(img, f"Frame: {result['frame']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 이미지 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        
        # 크기 조절
        label_w = self.image_label.width() - 10
        label_h = self.image_label.height() - 10
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        qimg = QImage(img_resized.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
    
    def on_inference_finished(self, results):
        """추론 완료"""
        self.tracking_results = results
        
        # UI 상태 복원
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.load_folder_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 일시정지")
        
        self.statusBar.showMessage(f"추론 완료! 총 {len(results)}개 탐지")
        
        QMessageBox.information(self, "완료", 
            f"추론이 완료되었습니다.\n\n"
            f"총 프레임: {len(self.image_files)}\n"
            f"탐지된 프레임: {len(results)}\n"
            f"탐지율: {len(results)/len(self.image_files)*100:.1f}%"
        )
    
    def on_inference_error(self, error_msg):
        """추론 에러"""
        QMessageBox.critical(self, "오류", f"추론 중 오류 발생:\n{error_msg}")
        self.on_inference_finished([])
    
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
        csv_path = output_dir / f"realtime_tracking_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # JSON 저장
        json_path = output_dir / f"realtime_tracking_{timestamp}.json"
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
        plot_path = output_dir / f"realtime_plot_{timestamp}.png"
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
        if self.inference_thread and self.inference_thread.isRunning():
            reply = QMessageBox.question(self, '확인', 
                "추론이 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.inference_thread.stop()
                self.inference_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    tracker = RealtimeTracker()
    tracker.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
