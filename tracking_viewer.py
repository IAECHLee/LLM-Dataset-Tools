"""
권취 줄 추적 결과 뷰어 - Object Detection 결과 확인용 GUI
"""
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QSpinBox,
    QCheckBox, QComboBox, QStatusBar, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor

# 기본 경로 설정
DEFAULT_IMAGE_DIR = r"K:\LLM Image_Storage\A line-2025-07-25_09-49-08(정상)"
DEFAULT_CSV_DIR = r"D:\LLM_Dataset\tracking_results"


def imread_korean(path):
    """한글 경로 이미지 로드"""
    stream = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


class TrackingViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("권취 줄 추적 결과 뷰어")
        self.setGeometry(100, 100, 1400, 900)
        
        # 데이터
        self.image_dir = None
        self.image_files = []
        self.tracking_df = None
        self.current_idx = 0
        self.show_bbox = True
        self.show_center = True
        self.show_trajectory = True
        self.auto_play = False
        self.play_speed = 100  # ms
        
        # 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_image)
        
        self.init_ui()
        self.load_default_data()
    
    def init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 왼쪽: 이미지 뷰어
        left_panel = QVBoxLayout()
        
        # 이미지 표시 영역
        self.image_label = QLabel()
        self.image_label.setMinimumSize(900, 700)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 2px solid #444;")
        left_panel.addWidget(self.image_label)
        
        # 슬라이더
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_change)
        slider_layout.addWidget(QLabel("프레임:"))
        slider_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(100)
        slider_layout.addWidget(self.frame_label)
        left_panel.addLayout(slider_layout)
        
        # 재생 컨트롤
        control_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀◀ 이전 (A)")
        self.prev_btn.clicked.connect(self.prev_image)
        control_layout.addWidget(self.prev_btn)
        
        self.play_btn = QPushButton("▶ 재생 (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)
        
        self.next_btn = QPushButton("다음 (D) ▶▶")
        self.next_btn.clicked.connect(self.next_image)
        control_layout.addWidget(self.next_btn)
        
        control_layout.addWidget(QLabel("속도:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(10, 1000)
        self.speed_spin.setValue(100)
        self.speed_spin.setSuffix(" ms")
        self.speed_spin.valueChanged.connect(self.on_speed_change)
        control_layout.addWidget(self.speed_spin)
        
        left_panel.addLayout(control_layout)
        
        main_layout.addLayout(left_panel, stretch=3)
        
        # 오른쪽: 정보 패널
        right_panel = QVBoxLayout()
        
        # 파일 로드
        load_group = QGroupBox("데이터 로드")
        load_layout = QVBoxLayout(load_group)
        
        self.load_csv_btn = QPushButton("CSV 파일 로드")
        self.load_csv_btn.clicked.connect(self.load_csv)
        load_layout.addWidget(self.load_csv_btn)
        
        self.load_img_btn = QPushButton("이미지 폴더 로드")
        self.load_img_btn.clicked.connect(self.load_image_folder)
        load_layout.addWidget(self.load_img_btn)
        
        self.csv_label = QLabel("CSV: 로드되지 않음")
        self.csv_label.setWordWrap(True)
        load_layout.addWidget(self.csv_label)
        
        self.img_label = QLabel("이미지: 로드되지 않음")
        self.img_label.setWordWrap(True)
        load_layout.addWidget(self.img_label)
        
        right_panel.addWidget(load_group)
        
        # 표시 옵션
        display_group = QGroupBox("표시 옵션")
        display_layout = QVBoxLayout(display_group)
        
        self.bbox_check = QCheckBox("Bounding Box 표시")
        self.bbox_check.setChecked(True)
        self.bbox_check.stateChanged.connect(self.on_display_change)
        display_layout.addWidget(self.bbox_check)
        
        self.center_check = QCheckBox("중심점 표시")
        self.center_check.setChecked(True)
        self.center_check.stateChanged.connect(self.on_display_change)
        display_layout.addWidget(self.center_check)
        
        self.traj_check = QCheckBox("궤적 표시 (최근 50프레임)")
        self.traj_check.setChecked(True)
        self.traj_check.stateChanged.connect(self.on_display_change)
        display_layout.addWidget(self.traj_check)
        
        right_panel.addWidget(display_group)
        
        # 현재 프레임 정보
        info_group = QGroupBox("현재 프레임 정보")
        info_layout = QVBoxLayout(info_group)
        
        self.info_filename = QLabel("파일: -")
        self.info_filename.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_filename)
        
        self.info_detection = QLabel("탐지: -")
        self.info_detection.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_detection)
        
        self.info_center = QLabel("중심점: -")
        self.info_center.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_center)
        
        self.info_bbox = QLabel("BBox: -")
        self.info_bbox.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_bbox)
        
        right_panel.addWidget(info_group)
        
        # 통계 정보
        stats_group = QGroupBox("전체 통계")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("데이터를 로드하세요")
        self.stats_label.setFont(QFont("Consolas", 9))
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        right_panel.addWidget(stats_group)
        
        # 이동 버튼
        nav_group = QGroupBox("빠른 이동")
        nav_layout = QVBoxLayout(nav_group)
        
        nav_btn_layout = QHBoxLayout()
        self.goto_first_det = QPushButton("첫 탐지")
        self.goto_first_det.clicked.connect(lambda: self.goto_detection('first'))
        nav_btn_layout.addWidget(self.goto_first_det)
        
        self.goto_last_det = QPushButton("마지막 탐지")
        self.goto_last_det.clicked.connect(lambda: self.goto_detection('last'))
        nav_btn_layout.addWidget(self.goto_last_det)
        nav_layout.addLayout(nav_btn_layout)
        
        frame_goto_layout = QHBoxLayout()
        frame_goto_layout.addWidget(QLabel("프레임 이동:"))
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(0, 0)
        frame_goto_layout.addWidget(self.goto_spin)
        self.goto_btn = QPushButton("이동")
        self.goto_btn.clicked.connect(self.goto_frame)
        frame_goto_layout.addWidget(self.goto_btn)
        nav_layout.addLayout(frame_goto_layout)
        
        right_panel.addWidget(nav_group)
        
        right_panel.addStretch()
        
        # 단축키 도움말
        help_group = QGroupBox("단축키")
        help_layout = QVBoxLayout(help_group)
        help_text = QLabel(
            "A / ← : 이전 프레임\n"
            "D / → : 다음 프레임\n"
            "Space : 재생/정지\n"
            "Home : 처음으로\n"
            "End : 마지막으로\n"
            "B : BBox 토글\n"
            "C : 중심점 토글\n"
            "T : 궤적 토글"
        )
        help_text.setFont(QFont("Consolas", 9))
        help_layout.addWidget(help_text)
        right_panel.addWidget(help_group)
        
        main_layout.addLayout(right_panel, stretch=1)
        
        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("준비")
    
    def keyPressEvent(self, event):
        """키보드 이벤트"""
        key = event.key()
        
        if key in [Qt.Key_A, Qt.Key_Left]:
            self.prev_image()
        elif key in [Qt.Key_D, Qt.Key_Right]:
            self.next_image()
        elif key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Home:
            self.goto_frame_idx(0)
        elif key == Qt.Key_End:
            self.goto_frame_idx(len(self.image_files) - 1)
        elif key == Qt.Key_B:
            self.bbox_check.setChecked(not self.bbox_check.isChecked())
        elif key == Qt.Key_C:
            self.center_check.setChecked(not self.center_check.isChecked())
        elif key == Qt.Key_T:
            self.traj_check.setChecked(not self.traj_check.isChecked())
    
    def load_default_data(self):
        """기본 데이터 로드"""
        # 가장 최근 CSV 파일 자동 로드
        csv_dir = Path(DEFAULT_CSV_DIR)
        if csv_dir.exists():
            csv_files = sorted(csv_dir.glob("tracking_*.csv"))
            if csv_files:
                self.load_csv_file(csv_files[-1])
        
        # 기본 이미지 폴더 로드
        img_dir = Path(DEFAULT_IMAGE_DIR)
        if img_dir.exists():
            self.load_image_folder_path(img_dir)
    
    def load_csv(self):
        """CSV 파일 선택 및 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "CSV 파일 선택", DEFAULT_CSV_DIR, "CSV Files (*.csv)"
        )
        if file_path:
            self.load_csv_file(Path(file_path))
    
    def load_csv_file(self, csv_path):
        """CSV 파일 로드"""
        try:
            self.tracking_df = pd.read_csv(csv_path)
            self.csv_label.setText(f"CSV: {csv_path.name}")
            self.update_stats()
            self.statusBar.showMessage(f"CSV 로드 완료: {len(self.tracking_df)}개 탐지 결과")
        except Exception as e:
            self.statusBar.showMessage(f"CSV 로드 실패: {e}")
    
    def load_image_folder(self):
        """이미지 폴더 선택"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "이미지 폴더 선택", DEFAULT_IMAGE_DIR
        )
        if folder_path:
            self.load_image_folder_path(Path(folder_path))
    
    def load_image_folder_path(self, folder_path):
        """이미지 폴더 로드"""
        self.image_dir = folder_path
        self.image_files = sorted([f for f in folder_path.glob("*.jpg")])
        
        if self.image_files:
            self.frame_slider.setMaximum(len(self.image_files) - 1)
            self.goto_spin.setMaximum(len(self.image_files) - 1)
            self.img_label.setText(f"이미지: {folder_path.name}\n({len(self.image_files)}장)")
            self.current_idx = 0
            self.update_display()
            self.statusBar.showMessage(f"이미지 로드 완료: {len(self.image_files)}장")
        else:
            self.statusBar.showMessage("이미지를 찾을 수 없습니다")
    
    def update_stats(self):
        """통계 업데이트"""
        if self.tracking_df is None or self.tracking_df.empty:
            return
        
        df = self.tracking_df
        stats_text = (
            f"총 탐지: {len(df)}개\n"
            f"프레임 범위: {df['frame'].min()} ~ {df['frame'].max()}\n"
            f"X 범위: {df['center_x'].min()} ~ {df['center_x'].max()}\n"
            f"Y 범위: {df['center_y'].min()} ~ {df['center_y'].max()}\n"
            f"X 평균: {df['center_x'].mean():.1f}\n"
            f"Y 평균: {df['center_y'].mean():.1f}"
        )
        self.stats_label.setText(stats_text)
    
    def get_detection_for_frame(self, frame_idx):
        """해당 프레임의 탐지 결과 반환"""
        if self.tracking_df is None:
            return None
        
        result = self.tracking_df[self.tracking_df['frame'] == frame_idx]
        if len(result) > 0:
            return result.iloc[0]
        return None
    
    def get_trajectory(self, frame_idx, window=50):
        """궤적 데이터 반환 (최근 window 프레임)"""
        if self.tracking_df is None:
            return []
        
        df = self.tracking_df
        recent = df[(df['frame'] <= frame_idx) & (df['frame'] > frame_idx - window)]
        return [(int(row['center_x']), int(row['center_y'])) for _, row in recent.iterrows()]
    
    def update_display(self):
        """화면 업데이트"""
        if not self.image_files or self.current_idx >= len(self.image_files):
            return
        
        # 이미지 로드
        img_path = self.image_files[self.current_idx]
        img = imread_korean(img_path)
        
        if img is None:
            self.statusBar.showMessage(f"이미지 로드 실패: {img_path.name}")
            return
        
        # 탐지 결과 가져오기
        detection = self.get_detection_for_frame(self.current_idx)
        
        # 그리기
        img_display = img.copy()
        
        # 궤적 그리기
        if self.traj_check.isChecked():
            trajectory = self.get_trajectory(self.current_idx)
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    alpha = i / len(trajectory)
                    color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                    cv2.line(img_display, trajectory[i-1], trajectory[i], color, 2)
        
        # 탐지 결과 그리기
        if detection is not None:
            cx, cy = int(detection['center_x']), int(detection['center_y'])
            bx, by = int(detection['bbox_x']), int(detection['bbox_y'])
            bw, bh = int(detection['bbox_w']), int(detection['bbox_h'])
            
            # Bounding Box
            if self.bbox_check.isChecked():
                cv2.rectangle(img_display, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
                cv2.putText(img_display, f"Class {int(detection['class_idx'])}", 
                           (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 중심점
            if self.center_check.isChecked():
                cv2.circle(img_display, (cx, cy), 10, (0, 0, 255), -1)
                cv2.circle(img_display, (cx, cy), 15, (255, 255, 255), 2)
                cv2.putText(img_display, f"({cx}, {cy})", 
                           (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 정보 업데이트
            self.info_detection.setText("탐지: ✓ 감지됨")
            self.info_detection.setStyleSheet("color: green;")
            self.info_center.setText(f"중심점: ({cx}, {cy})")
            self.info_bbox.setText(f"BBox: ({bx}, {by}, {bw}, {bh})")
        else:
            self.info_detection.setText("탐지: ✗ 미감지")
            self.info_detection.setStyleSheet("color: red;")
            self.info_center.setText("중심점: -")
            self.info_bbox.setText("BBox: -")
        
        # 프레임 정보 표시
        cv2.putText(img_display, f"Frame: {self.current_idx}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 이미지 변환 및 표시
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
        
        # UI 업데이트
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_idx)
        self.frame_slider.blockSignals(False)
        
        self.frame_label.setText(f"{self.current_idx} / {len(self.image_files) - 1}")
        self.info_filename.setText(f"파일: {img_path.name}")
    
    def on_slider_change(self, value):
        """슬라이더 변경"""
        self.current_idx = value
        self.update_display()
    
    def on_display_change(self):
        """표시 옵션 변경"""
        self.update_display()
    
    def on_speed_change(self, value):
        """재생 속도 변경"""
        self.play_speed = value
        if self.auto_play:
            self.timer.setInterval(value)
    
    def prev_image(self):
        """이전 이미지"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_image(self):
        """다음 이미지"""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.update_display()
        elif self.auto_play:
            self.toggle_play()
    
    def toggle_play(self):
        """재생/정지 토글"""
        self.auto_play = not self.auto_play
        if self.auto_play:
            self.play_btn.setText("⏸ 정지 (Space)")
            self.timer.start(self.play_speed)
        else:
            self.play_btn.setText("▶ 재생 (Space)")
            self.timer.stop()
    
    def goto_detection(self, which):
        """탐지된 프레임으로 이동"""
        if self.tracking_df is None:
            return
        
        if which == 'first':
            frame = self.tracking_df['frame'].min()
        else:
            frame = self.tracking_df['frame'].max()
        
        self.goto_frame_idx(frame)
    
    def goto_frame(self):
        """지정 프레임으로 이동"""
        self.goto_frame_idx(self.goto_spin.value())
    
    def goto_frame_idx(self, idx):
        """프레임 인덱스로 이동"""
        if 0 <= idx < len(self.image_files):
            self.current_idx = idx
            self.update_display()


def main():
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    viewer = TrackingViewer()
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
