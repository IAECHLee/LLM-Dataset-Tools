#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
균일 인덱스 기반 이미지 샘플러 (Uniform Index-based Image Sampler)
layer_sampler.py 기반으로 구현
- 키워드로 폴더 검색
- 폴더명에서 (정상), (불량) 키워드로 분류
- 레이어 정보 없이 전체 이미지에서 균일한 간격으로 샘플링
예: 3600장에서 1% = 36장 → 인덱스 0, 100, 200, 300... 방식으로 추출
"""
import sys
import os
import re
import json
import shutil
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QSplitter,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox,
    QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QImage, QFont, QWheelEvent
import numpy as np
import cv2

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def natural_key(s: str):
    """자연 정렬을 위한 키 함수"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def list_images(folder: Path):
    """폴더 내 이미지 파일 목록 반환 (자연 정렬)"""
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def find_target_folders(root: Path, keywords: list):
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


class UniformSamplerThread(QThread):
    """균일 인덱스 샘플링 스레드 (랜덤 분포 지원)"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)  # 결과 통계
    error = pyqtSignal(str)
    preview_image = pyqtSignal(str)  # 이미지 경로

    def __init__(self, root_folder, output_folder, keywords, sample_percent, seed=42):
        super().__init__()
        self.root_folder = Path(root_folder)
        self.output_folder = Path(output_folder)
        self.keywords = keywords
        self.sample_percent = sample_percent
        self.seed = seed
        self.is_cancelled = False
        self.folder_count = 0  # 폴더별 다른 오프셋을 위한 카운터
        import random
        self.rng = random.Random(seed)

    def cancel(self):
        self.is_cancelled = True

    def get_distributed_indices(self, total_count, sample_count, folder_index, total_folders):
        """
        폴더별로 다른 시작점을 가진 균일 분포 인덱스 계산
        각 폴더가 다른 위치의 이미지를 샘플링하여 전체적으로 골고루 분포되게 함
        
        Args:
            total_count: 폴더 내 전체 이미지 수
            sample_count: 샘플링할 이미지 수
            folder_index: 현재 폴더의 인덱스 (0부터 시작)
            total_folders: 전체 폴더 수
        """
        if sample_count >= total_count:
            return list(range(total_count))
        elif sample_count == 1:
            # 폴더마다 다른 위치에서 1장 선택
            offset = int((total_count / total_folders) * folder_index) if total_folders > 0 else 0
            return [offset % total_count]
        else:
            # 기본 간격 계산
            interval = total_count / sample_count
            
            # 폴더별로 다른 시작 오프셋 (전체 간격을 폴더 수로 나눈 만큼씩 이동)
            if total_folders > 1:
                phase_offset = (interval / total_folders) * folder_index
            else:
                phase_offset = 0
            
            # 각 샘플 위치에 약간의 랜덤 지터 추가 (간격의 ±30% 범위)
            jitter_range = interval * 0.3
            
            indices = []
            for i in range(sample_count):
                # 기본 위치 + 폴더별 오프셋 + 랜덤 지터
                base_pos = i * interval + phase_offset
                jitter = self.rng.uniform(-jitter_range, jitter_range)
                pos = int(round(base_pos + jitter))
                
                # 범위 제한
                pos = max(0, min(total_count - 1, pos))
                indices.append(pos)
            
            # 중복 제거 및 정렬
            return sorted(set(indices))

    def process_folder(self, folder: Path, sample_ratio: float, folder_index: int, total_folders: int):
        """
        폴더에서 이미지를 분산 인덱스로 샘플링
        폴더명에 (정상) 또는 (불량)이 포함되어 있어야 함
        
        Returns:
            {"normal": [image_paths], "defect": [image_paths]}
        """
        result = {"normal": [], "defect": []}
        
        # 폴더명으로 정상/불량 판단
        is_normal = "(정상)" in folder.name
        is_defect = "(불량)" in folder.name
        
        if not is_normal and not is_defect:
            return result
        
        category = "normal" if is_normal else "defect"
        
        # 폴더 내 이미지 읽기
        images = list_images(folder)
        if not images:
            return result
        
        total_count = len(images)
        sample_count = max(1, int(total_count * sample_ratio))
        
        # 분산 인덱스 계산 (폴더별로 다른 위치에서 샘플링)
        selected_indices = self.get_distributed_indices(total_count, sample_count, folder_index, total_folders)
        
        # 선택된 이미지 추가
        result[category] = [images[idx] for idx in selected_indices]
        
        return result

    def run(self):
        try:
            # 루트 폴더 확인
            if not self.root_folder.exists():
                self.error.emit(f"루트 폴더를 찾을 수 없습니다: {self.root_folder}")
                return
            
            self.log.emit(f"[INFO] 루트 디렉토리: {self.root_folder}")
            self.log.emit(f"[INFO] 검색 키워드: {self.keywords}")
            self.log.emit(f"[INFO] 샘플링 비율: {self.sample_percent}%")
            
            # 키워드로 폴더 찾기
            target_folders = find_target_folders(self.root_folder, self.keywords)
            
            if not target_folders:
                self.error.emit(f"키워드 {self.keywords}를 모두 포함하는 폴더를 찾을 수 없습니다.")
                return
            
            self.log.emit(f"\n[INFO] 발견된 폴더: {len(target_folders)}개")
            for folder in target_folders:
                self.log.emit(f"  - {folder.name}")
            
            # 출력 폴더 구조 생성
            normal_dir = self.output_folder / "Normal"
            defect_dir = self.output_folder / "Defect"
            normal_dir.mkdir(parents=True, exist_ok=True)
            defect_dir.mkdir(parents=True, exist_ok=True)
            
            # 통계 정보
            total_stats = {"normal": 0, "defect": 0}
            manifest = []
            sample_ratio = self.sample_percent / 100.0
            
            # 전체 작업량 추정
            total_folders = len(target_folders)
            
            # 각 폴더 처리
            for folder_idx, folder in enumerate(target_folders):
                if self.is_cancelled:
                    self.log.emit("[WARN] 사용자에 의해 취소되었습니다.")
                    break
                
                self.log.emit(f"\n[INFO] 처리 중 ({folder_idx+1}/{total_folders}): {folder.name}")
                self.progress.emit(folder_idx + 1, total_folders, f"폴더 처리 중: {folder.name}")
                
                # 분산 인덱스로 이미지 샘플링 (폴더 인덱스 전달)
                sampled_images = self.process_folder(folder, sample_ratio, folder_idx, total_folders)
                
                # Normal 이미지 복사
                if sampled_images["normal"]:
                    for img_path in sampled_images["normal"]:
                        dst_name = f"{folder.name}_{img_path.name}"
                        dst_path = normal_dir / dst_name
                        shutil.copy2(img_path, dst_path)
                        manifest.append({
                            "source": str(img_path),
                            "destination": str(dst_path),
                            "folder": folder.name,
                            "category": "normal",
                            "filename": img_path.name
                        })
                        total_stats["normal"] += 1
                        
                        # 미리보기 (10장마다)
                        if total_stats["normal"] % 10 == 1:
                            self.preview_image.emit(str(img_path))
                    
                    self.log.emit(f"  - Normal: {len(sampled_images['normal'])}개 복사")
                
                # Defect 이미지 복사
                if sampled_images["defect"]:
                    for img_path in sampled_images["defect"]:
                        dst_name = f"{folder.name}_{img_path.name}"
                        dst_path = defect_dir / dst_name
                        shutil.copy2(img_path, dst_path)
                        manifest.append({
                            "source": str(img_path),
                            "destination": str(dst_path),
                            "folder": folder.name,
                            "category": "defect",
                            "filename": img_path.name
                        })
                        total_stats["defect"] += 1
                        
                        # 미리보기 (10장마다)
                        if total_stats["defect"] % 10 == 1:
                            self.preview_image.emit(str(img_path))
                    
                    self.log.emit(f"  - Defect: {len(sampled_images['defect'])}개 복사")
            
            # Manifest 저장
            manifest_path = self.output_folder / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            # 통계 저장
            stats_path = self.output_folder / "stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(total_stats, f, indent=2, ensure_ascii=False)
            
            # 결과 출력
            self.log.emit(f"\n{'='*60}")
            self.log.emit(f"[완료] 균일 인덱스 샘플링 완료!")
            self.log.emit(f"{'='*60}")
            self.log.emit(f"\n정상 이미지 (Normal): {total_stats['normal']}개")
            self.log.emit(f"불량 이미지 (Defect): {total_stats['defect']}개")
            self.log.emit(f"총 복사된 이미지: {total_stats['normal'] + total_stats['defect']}개")
            self.log.emit(f"\n출력 폴더: {self.output_folder}")
            self.log.emit(f"  - Normal: {normal_dir}")
            self.log.emit(f"  - Defect: {defect_dir}")
            self.log.emit(f"  - Manifest: {manifest_path}")
            self.log.emit(f"  - 통계: {stats_path}")
            
            self.finished_signal.emit(total_stats)
            
        except Exception as e:
            import traceback
            self.error.emit(f"오류 발생: {str(e)}\n{traceback.format_exc()}")


class ZoomableGraphicsView(QGraphicsView):
    """마우스 휠로 확대/축소 가능한 GraphicsView"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._zoom = 1.0
        
    def wheelEvent(self, event: QWheelEvent):
        """마우스 휠로 확대/축소"""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self._zoom *= factor
            self.scale(factor, factor)
        else:
            self._zoom /= factor
            self.scale(1/factor, 1/factor)
            
    def mouseDoubleClickEvent(self, event):
        """더블클릭으로 원본 크기로 복원"""
        self.resetTransform()
        self._zoom = 1.0
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)


class ImageViewer(QWidget):
    """이미지 뷰어 위젯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_image_path = None
        self.rotation = 0
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 이미지 정보 라벨
        self.info_label = QLabel("이미지를 선택하세요")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #aaa; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # GraphicsView
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)
        self.view.setMinimumSize(400, 300)
        layout.addWidget(self.view)
        
        # 회전 버튼
        btn_layout = QHBoxLayout()
        self.rotate_btn = QPushButton("🔄 90° 회전")
        self.rotate_btn.clicked.connect(self.rotate_image)
        btn_layout.addStretch()
        btn_layout.addWidget(self.rotate_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
    def load_image(self, image_path: str):
        """이미지 로드 (한글 경로 지원)"""
        self.current_image_path = image_path
        self.rotation = 0
        
        try:
            # 한글 경로 지원
            img_array = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                self.info_label.setText(f"이미지 로드 실패: {Path(image_path).name}")
                return
            
            self._display_cv_image(img)
            
            # 이미지 정보 표시
            h, w = img.shape[:2]
            filename = Path(image_path).name
            self.info_label.setText(f"{filename} ({w}x{h})")
            
        except Exception as e:
            self.info_label.setText(f"오류: {str(e)}")
    
    def _display_cv_image(self, img):
        """OpenCV 이미지를 화면에 표시"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        
        img_contiguous = np.ascontiguousarray(img_rgb)
        qimg = QImage(img_contiguous.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
    def rotate_image(self):
        """이미지 90도 회전"""
        if not self.current_image_path:
            return
            
        self.rotation = (self.rotation + 90) % 360
        
        try:
            img_array = np.fromfile(self.current_image_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return
            
            # 회전 적용
            if self.rotation == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif self.rotation == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            self._display_cv_image(img)
            
        except Exception as e:
            self.info_label.setText(f"회전 오류: {str(e)}")


class UniformSamplerGUI(QMainWindow):
    """균일 인덱스 샘플러 GUI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("균일 인덱스 샘플러 (Uniform Index Sampler)")
        self.setGeometry(100, 100, 1200, 800)
        self.sampler_thread = None
        self.setup_ui()
        self.apply_dark_theme()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # 왼쪽 패널 - 설정 및 로그
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 루트 폴더 설정
        root_group = QGroupBox("루트 폴더")
        root_layout = QHBoxLayout(root_group)
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("이미지가 있는 루트 폴더 선택...")
        self.root_edit.setText(r"K:\LLM Image_Storage")
        root_btn = QPushButton("찾아보기")
        root_btn.clicked.connect(self.browse_root)
        root_layout.addWidget(self.root_edit)
        root_layout.addWidget(root_btn)
        left_layout.addWidget(root_group)
        
        # 키워드 설정
        keyword_group = QGroupBox("검색 키워드 (쉼표로 구분)")
        keyword_layout = QHBoxLayout(keyword_group)
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("예: A line, 2025-07-27")
        self.keyword_edit.setText("A line, 2025-07-27")
        keyword_layout.addWidget(self.keyword_edit)
        left_layout.addWidget(keyword_group)
        
        # 출력 폴더 설정
        output_group = QGroupBox("출력 폴더")
        output_layout = QHBoxLayout(output_group)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("샘플링된 이미지를 저장할 폴더...")
        self.output_edit.setText(r"D:\LLM_Dataset\output\Uniform Sample")
        output_btn = QPushButton("찾아보기")
        output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_btn)
        left_layout.addWidget(output_group)
        
        # 샘플링 설정
        sample_group = QGroupBox("샘플링 설정")
        sample_layout = QHBoxLayout(sample_group)
        
        sample_layout.addWidget(QLabel("샘플링 비율:"))
        self.percent_spin = QDoubleSpinBox()
        self.percent_spin.setRange(0.1, 100.0)
        self.percent_spin.setValue(5.0)
        self.percent_spin.setSuffix(" %")
        self.percent_spin.setDecimals(1)
        sample_layout.addWidget(self.percent_spin)
        
        sample_layout.addStretch()
        
        sample_layout.addWidget(QLabel("랜덤 시드:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        sample_layout.addWidget(self.seed_spin)
        
        left_layout.addWidget(sample_group)
        
        # 실행 버튼
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ 샘플링 시작")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_sampling)
        btn_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("⏹ 취소")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_sampling)
        btn_layout.addWidget(self.cancel_btn)
        left_layout.addLayout(btn_layout)
        
        # 진행률
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        left_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("대기 중...")
        left_layout.addWidget(self.progress_label)
        
        # 로그
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_edit)
        left_layout.addWidget(log_group)
        
        # 오른쪽 패널 - 이미지 미리보기
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        preview_group = QGroupBox("이미지 미리보기")
        preview_layout = QVBoxLayout(preview_group)
        self.image_viewer = ImageViewer()
        preview_layout.addWidget(self.image_viewer)
        right_layout.addWidget(preview_group)
        
        # 스플리터로 좌우 패널 배치
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])
        main_layout.addWidget(splitter)
        
        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("준비됨")
        
    def apply_dark_theme(self):
        """다크 테마 적용"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #3c3c3c;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 5px;
            }
        """)
        
    def browse_root(self):
        """루트 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "루트 폴더 선택", r"K:\LLM Image_Storage")
        if folder:
            self.root_edit.setText(folder)
            
    def browse_output(self):
        """출력 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "출력 폴더 선택", r"D:\LLM_Dataset\output")
        if folder:
            self.output_edit.setText(folder)
            
    def start_sampling(self):
        """샘플링 시작"""
        root = self.root_edit.text().strip()
        output = self.output_edit.text().strip()
        keywords_text = self.keyword_edit.text().strip()
        
        if not root:
            QMessageBox.warning(self, "경고", "루트 폴더를 선택하세요.")
            return
            
        if not Path(root).exists():
            QMessageBox.warning(self, "경고", "루트 폴더가 존재하지 않습니다.")
            return
            
        if not output:
            QMessageBox.warning(self, "경고", "출력 폴더를 지정하세요.")
            return
        
        if not keywords_text:
            QMessageBox.warning(self, "경고", "검색 키워드를 입력하세요.")
            return
        
        # 키워드 파싱
        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
        if not keywords:
            QMessageBox.warning(self, "경고", "유효한 검색 키워드를 입력하세요.")
            return
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_edit.clear()
        
        # 스레드 시작
        self.sampler_thread = UniformSamplerThread(
            root, output, keywords,
            self.percent_spin.value(),
            self.seed_spin.value()
        )
        self.sampler_thread.progress.connect(self.on_progress)
        self.sampler_thread.log.connect(self.on_log)
        self.sampler_thread.finished_signal.connect(self.on_finished)
        self.sampler_thread.error.connect(self.on_error)
        self.sampler_thread.preview_image.connect(self.on_preview)
        self.sampler_thread.start()
        
    def cancel_sampling(self):
        """샘플링 취소"""
        if self.sampler_thread:
            self.sampler_thread.cancel()
            
    def on_progress(self, current, total, message):
        """진행률 업데이트"""
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"{current}/{total} - {message}")
        
    def on_log(self, message):
        """로그 추가"""
        self.log_edit.append(message)
        
    def on_preview(self, image_path):
        """이미지 미리보기"""
        self.image_viewer.load_image(image_path)
        
    def on_finished(self, stats):
        """완료 처리"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        total_count = stats.get('normal', 0) + stats.get('defect', 0)
        self.statusBar.showMessage(f"완료: {total_count}장 샘플링됨 (Normal: {stats.get('normal', 0)}, Defect: {stats.get('defect', 0)})")
        QMessageBox.information(self, "완료", 
            f"균일 인덱스 샘플링 완료!\n\n"
            f"Normal: {stats.get('normal', 0)}장\n"
            f"Defect: {stats.get('defect', 0)}장\n"
            f"총: {total_count}장")
        
    def on_error(self, error_msg):
        """오류 처리"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.statusBar.showMessage(f"오류: {error_msg}")
        QMessageBox.critical(self, "오류", error_msg)


def main():
    app = QApplication(sys.argv)
    window = UniformSamplerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
