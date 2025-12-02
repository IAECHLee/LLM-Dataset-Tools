#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer Sampler GUI - PyQt5 기반 이미지 샘플링 도구

주요 기능:
1. 전체 샘플링: 모든 레이어에서 지정된 비율로 이미지 샘플링
2. 부분 샘플링: 특정 레이어/카테고리에서 추가 샘플링 (중복 방지)
3. 이미지 뷰어: Normal/Defect 이미지 확인 및 분류 수정
4. 랜덤 삭제: 목표 개수에 맞춰 이미지 랜덤 삭제
5. 샘플링 히스토리: 이미 선택된 이미지 자동 추적 및 중복 방지
"""
import sys
import json
import shutil
import random
import time
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTextEdit, QFileDialog, QProgressBar, QGroupBox, QGridLayout,
    QListWidget, QSplitter, QMessageBox, QTabWidget, QShortcut,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QKeySequence, QPainter, QBrush, QColor
import layer_sampler as ls


class SamplerThread(QThread):
    """백그라운드에서 샘플링 작업을 수행하는 스레드"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, root, out, layers_file, sample_ratio, seed, keywords, 
                 sampling_history=None, target_layer=None, target_category=None):
        super().__init__()
        self.root = Path(root)
        self.out = Path(out)
        self.layers_file = Path(layers_file)
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.keywords = keywords
        self.sampling_history = sampling_history or {}  # 이미 샘플링된 이미지 히스토리
        self.target_layer = target_layer  # None이면 모든 레이어, 숫자면 특정 레이어만
        self.target_category = target_category  # None: both, 0: normal, 1: defect
        
    def run(self):
        try:
            result = self.process_sampling()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def process_sampling(self):
        """이미지 샘플링 처리"""
        normal_dir = self.out / "Normal Layer"
        defect_dir = self.out / "Defect Layer"
        ls.ensure_dir(normal_dir)
        ls.ensure_dir(defect_dir)
        
        # 레이어 파일 읽기
        if not self.layers_file.exists():
            raise FileNotFoundError(f"레이어 파일을 찾을 수 없습니다: {self.layers_file}")
        
        with open(self.layers_file, 'r', encoding='utf-8') as f:
            layer_data = json.load(f)
        
        first_winding_id = list(layer_data.keys())[0]
        layers = [[layer['start'], layer['end'], layer['count']] for layer in layer_data[first_winding_id]]
        num_layers = len(layers)
        
        self.progress.emit(f"레이어 정보: {num_layers}개 층 ({first_winding_id})")
        self.progress.emit(f"루트 디렉토리: {self.root}")
        self.progress.emit(f"검색 키워드: {self.keywords}")
        self.progress.emit(f"샘플링 비율: {self.sample_ratio * 100:.1f}%\n")
        
        if not self.root.exists():
            raise FileNotFoundError(f"루트 디렉토리를 찾을 수 없습니다: {self.root}")
        
        # 폴더 찾기
        target_folders = ls.find_target_folders(self.root, self.keywords)
        
        if not target_folders:
            raise ValueError(f"키워드 {self.keywords}를 모두 포함하는 폴더를 찾을 수 없습니다")
        
        self.progress.emit(f"발견된 폴더: {len(target_folders)}개")
        for folder in target_folders:
            self.progress.emit(f"  - {folder.name}")
        
        # 통계 정보
        total_stats = {
            "normal": {layer_id: 0 for layer_id in range(1, num_layers + 1)},
            "defect": {layer_id: 0 for layer_id in range(1, num_layers + 1)}
        }
        manifest = []
        
        # 각 폴더 처리 (각 폴더마다 다른 시드 사용)
        for folder_idx, folder in enumerate(target_folders):
            self.progress.emit(f"\n처리 중: {folder.name}")
            
            # 폴더마다 완전히 다른 시드: 기본 시드 + 폴더 이름 해시 + 인덱스
            folder_hash = hash(folder.name) % 10000
            folder_seed = self.seed + folder_hash + (folder_idx * 1000)
            
            # 이미 샘플링된 이미지 목록 가져오기
            folder_history = self.sampling_history.get(folder.name, [])
            
            sampled_images = ls.process_folder_with_layers(
                folder, layers, self.sample_ratio, folder_seed, 
                exclude_images=set(folder_history)
            )
            
            # Normal 이미지 복사 (target_category 필터링)
            if self.target_category in [None, 0]:  # None: both, 0: normal
                for layer_id, image_paths in sampled_images["normal"].items():
                    # target_layer 필터링
                    if self.target_layer is not None and self.target_layer != 0 and layer_id != self.target_layer:
                        continue
                    
                    if not image_paths:
                        continue
                    layer_folder = normal_dir / f"Layer_{layer_id:02d}"
                    ls.ensure_dir(layer_folder)
                    
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
                        
                        # 히스토리에 추가
                        if folder.name not in self.sampling_history:
                            self.sampling_history[folder.name] = []
                        self.sampling_history[folder.name].append(img_path.name)
                    
                    self.progress.emit(f"  - Normal Layer {layer_id:02d}: {len(image_paths)}개 복사")
            
            # Defect 이미지 복사 (target_category 필터링)
            if self.target_category in [None, 1]:  # None: both, 1: defect
                for layer_id, image_paths in sampled_images["defect"].items():
                    # target_layer 필터링
                    if self.target_layer is not None and self.target_layer != 0 and layer_id != self.target_layer:
                        continue
                    
                    if not image_paths:
                        continue
                    layer_folder = defect_dir / f"Layer_{layer_id:02d}"
                    ls.ensure_dir(layer_folder)
                    
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
                        
                        # 히스토리에 추가
                        if folder.name not in self.sampling_history:
                            self.sampling_history[folder.name] = []
                        self.sampling_history[folder.name].append(img_path.name)
                    
                    self.progress.emit(f"  - Defect Layer {layer_id:02d}: {len(image_paths)}개 복사")
        
        # Manifest 저장
        manifest_path = self.out / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        # 통계 저장
        stats_path = self.out / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(total_stats, f, indent=2, ensure_ascii=False)
        
        return {
            "stats": total_stats,
            "manifest": manifest,
            "num_layers": num_layers,
            "output_dir": str(self.out),
            "sampling_history": self.sampling_history  # 업데이트된 히스토리 반환
        }


class ZoomableGraphicsView(QGraphicsView):
    """스크롤 확대/축소가 가능한 QGraphicsView"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(43, 43, 43)))
        self._zoom = 0
        self._min_zoom = -5
        self._max_zoom = 10
    
    def wheelEvent(self, event):
        """마우스 휠 이벤트: 확대/축소"""
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        
        if self._zoom > self._max_zoom:
            self._zoom = self._max_zoom
            return
        if self._zoom < self._min_zoom:
            self._zoom = self._min_zoom
            return
        
        self.scale(factor, factor)
    
    def reset_zoom(self):
        """확대/축소 초기화"""
        self.resetTransform()
        self._zoom = 0


class ImageViewer(QWidget):
    """이미지 뷰어 위젯 (스크롤 확대/축소 지원)"""
    def __init__(self, title="이미지"):
        super().__init__()
        self.title = title
        self.current_image_path = None
        self.scene = QGraphicsScene()
        self.pixmap_item = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 이미지 표시 뷰어 (스크롤 확대/축소 가능)
        self.graphics_view = ZoomableGraphicsView()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setMinimumSize(400, 400)
        self.graphics_view.setStyleSheet("QGraphicsView { background-color: #2b2b2b; border: 2px solid #555; }")
        
        # 이미지 정보
        self.info_label = QLabel("파일: - | 마우스 휠: 확대/축소 | 드래그: 이동 | 더블클릭: 초기화")
        self.info_label.setStyleSheet("QLabel { padding: 5px; font-size: 9pt; }")
        self.info_label.setWordWrap(True)
        
        layout.addWidget(self.graphics_view, 1)
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        
        # 더블클릭 이벤트로 확대/축소 초기화
        self.graphics_view.mouseDoubleClickEvent = self.on_double_click
    
    def on_double_click(self, event):
        """더블클릭 시 확대/축소 초기화"""
        self.graphics_view.reset_zoom()
        if self.pixmap_item:
            self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
    
    def display_image(self, image_path: Path):
        """이미지 표시"""
        if not image_path or not image_path.exists():
            self.scene.clear()
            self.pixmap_item = None
            self.current_image_path = None
            self.info_label.setText("이미지를 찾을 수 없습니다")
            return
        
        try:
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                self.scene.clear()
                self.pixmap_item = None
                self.current_image_path = None
                self.info_label.setText("이미지를 로드할 수 없습니다")
                return
            
            # Scene 초기화 및 이미지 추가
            self.scene.clear()
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            
            # 오른쪽으로 90도 회전 (시계방향)
            # 이미지 중심을 기준으로 회전하기 위해 transform origin 설정
            self.pixmap_item.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
            self.pixmap_item.setRotation(90)
            
            self.scene.addItem(self.pixmap_item)
            self.scene.setSceneRect(self.pixmap_item.sceneBoundingRect())
            
            # 초기 뷰 설정: 이미지가 뷰에 맞도록
            self.graphics_view.reset_zoom()
            self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            
            self.current_image_path = image_path
            
            # 이미지 정보 표시
            self.info_label.setText(
                f"파일: {image_path.name} | "
                f"크기: {pixmap.width()}x{pixmap.height()} | "
                f"용량: {image_path.stat().st_size / 1024:.1f} KB | "
                f"마우스 휠: 확대/축소 | 드래그: 이동 | 더블클릭: 초기화"
            )
        except Exception as e:
            self.scene.clear()
            self.pixmap_item = None
            self.current_image_path = None
            self.info_label.setText(f"오류: {str(e)}")
    
    def clear(self):
        """이미지 초기화"""
        self.scene.clear()
        self.pixmap_item = None
        self.current_image_path = None
        self.info_label.setText("파일: - | 마우스 휠: 확대/축소 | 드래그: 이동 | 더블클릭: 초기화")


class LayerSamplerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sampler_thread = None
        self.current_output_dir = None
        self.normal_images = {}  # {layer_id: [image_paths]}
        self.defect_images = {}  # {layer_id: [image_paths]}
        self.last_focused_category = "normal"  # 마지막으로 포커스된 카테고리
        self.sampling_history_file = Path("sampling_history.json")  # 샘플링 히스토리 파일
        self.last_position_file = Path("last_position.json")  # 마지막 선택 위치 저장
        self.undo_stack = []  # 실행 취소 스택 (최대 5개)
        self.max_undo = 5
        self.init_ui()
        self.setup_shortcuts()
        
    def init_ui(self):
        self.setWindowTitle("Layer Sampler - 이미지 샘플링 도구")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        
        # 왼쪽: 설정 패널
        left_panel = self.create_settings_panel()
        
        # 오른쪽: 탭 (로그 + 이미지 뷰어)
        right_panel = self.create_right_panel()
        
        # 스플리터로 분할
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        
        # 스타일 적용
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
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
                color: #d4d4d4;
            }
            QTextEdit, QListWidget {
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
        """)
    
    def create_settings_panel(self):
        """설정 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 입력 설정
        input_group = QGroupBox("입력 설정")
        input_layout = QGridLayout()
        
        # 루트 디렉토리
        input_layout.addWidget(QLabel("루트 디렉토리:"), 0, 0)
        self.root_input = QLineEdit(r"K:\LLM Image_Storage")
        input_layout.addWidget(self.root_input, 0, 1)
        root_browse_btn = QPushButton("찾아보기")
        root_browse_btn.clicked.connect(self.browse_root)
        input_layout.addWidget(root_browse_btn, 0, 2)
        
        # 레이어 파일
        input_layout.addWidget(QLabel("레이어 파일:"), 1, 0)
        self.layers_input = QLineEdit("layers.json")
        input_layout.addWidget(self.layers_input, 1, 1)
        layers_browse_btn = QPushButton("찾아보기")
        layers_browse_btn.clicked.connect(self.browse_layers)
        input_layout.addWidget(layers_browse_btn, 1, 2)
        
        # 출력 디렉토리
        input_layout.addWidget(QLabel("출력 디렉토리:"), 2, 0)
        self.out_input = QLineEdit(r"D:\LLM_Dataset\output")
        input_layout.addWidget(self.out_input, 2, 1)
        out_browse_btn = QPushButton("찾아보기")
        out_browse_btn.clicked.connect(self.browse_output)
        input_layout.addWidget(out_browse_btn, 2, 2)
        
        input_group.setLayout(input_layout)
        
        # 샘플링 설정
        sampling_group = QGroupBox("샘플링 설정")
        sampling_layout = QGridLayout()
        
        # 샘플링 비율
        sampling_layout.addWidget(QLabel("샘플링 비율 (%):"), 0, 0)
        self.ratio_input = QDoubleSpinBox()
        self.ratio_input.setRange(0.1, 100.0)
        self.ratio_input.setValue(5.0)
        self.ratio_input.setSingleStep(0.5)
        self.ratio_input.setSuffix(" %")
        sampling_layout.addWidget(self.ratio_input, 0, 1)
        
        # 검색 키워드
        sampling_layout.addWidget(QLabel("검색 키워드:"), 1, 0)
        self.keywords_input = QLineEdit("A line, 2025-07-27")
        sampling_layout.addWidget(self.keywords_input, 1, 1)
        
        sampling_group.setLayout(sampling_layout)
        
        # 실행 버튼
        self.run_btn = QPushButton("샘플링 시작")
        self.run_btn.clicked.connect(self.start_sampling)
        self.run_btn.setMinimumHeight(40)
        
        # 이미지 뷰어 열기 버튼
        self.open_viewer_btn = QPushButton("이미지 뷰어 열기")
        self.open_viewer_btn.clicked.connect(self.open_image_viewer)
        self.open_viewer_btn.setMinimumHeight(40)
        self.open_viewer_btn.setStyleSheet("""
            QPushButton {
                background-color: #0a7a0a;
            }
            QPushButton:hover {
                background-color: #0c9a0c;
            }
            QPushButton:pressed {
                background-color: #086308;
            }
        """)
        
        # 부분 샘플링 버튼
        self.partial_sampling_btn = QPushButton("부분 샘플링")
        self.partial_sampling_btn.clicked.connect(self.open_partial_sampling_dialog)
        self.partial_sampling_btn.setMinimumHeight(40)
        self.partial_sampling_btn.setStyleSheet("""
            QPushButton {
                background-color: #6a5acd;
            }
            QPushButton:hover {
                background-color: #7b68ee;
            }
            QPushButton:pressed {
                background-color: #5a4abd;
            }
        """)
        
        # 샘플링 히스토리 관리 버튼들
        history_layout = QHBoxLayout()
        
        self.clear_history_btn = QPushButton("리스트 클린")
        self.clear_history_btn.clicked.connect(self.clear_sampling_history)
        self.clear_history_btn.setToolTip("샘플링 히스토리를 삭제하고 새로 시작합니다")
        self.clear_history_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
            QPushButton:pressed {
                background-color: #c62828;
            }
        """)
        history_layout.addWidget(self.clear_history_btn)
        
        self.view_history_btn = QPushButton("히스토리 보기")
        self.view_history_btn.clicked.connect(self.view_sampling_history)
        self.view_history_btn.setToolTip("현재 샘플링 히스토리 정보를 확인합니다")
        history_layout.addWidget(self.view_history_btn)
        
        # Undo 스택 정보 레이블
        self.undo_label = QLabel("실행 취소 가능: 0개 (Ctrl+Z)")
        self.undo_label.setStyleSheet("QLabel { color: #888; font-size: 9pt; padding: 5px; }")
        self.undo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.undo_label)
        
        # 진행률
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 레이아웃 구성
        layout.addWidget(input_group)
        layout.addWidget(sampling_group)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.open_viewer_btn)
        layout.addWidget(self.partial_sampling_btn)
        layout.addLayout(history_layout)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self):
        """오른쪽 패널 (탭) 생성"""
        tab_widget = QTabWidget()
        
        # 로그 탭
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_tab.setLayout(log_layout)
        
        # 듀얼 이미지 뷰어 탭
        viewer_tab = QWidget()
        viewer_layout = QVBoxLayout()
        
        # 레이어 선택 및 제어 컨트롤
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("레이어 선택:"))
        
        self.layer_selector = QSpinBox()
        self.layer_selector.setRange(1, 19)
        self.layer_selector.setValue(1)
        self.layer_selector.setPrefix("Layer ")
        self.layer_selector.valueChanged.connect(self.on_layer_changed)
        control_layout.addWidget(self.layer_selector)
        
        control_layout.addSpacing(20)
        
        # Normal Layer 이미지 개수 정보
        control_layout.addWidget(QLabel("Normal:"))
        self.normal_count_label = QLabel("0개")
        self.normal_count_label.setStyleSheet("QLabel { font-weight: bold; color: #4a9eff; }")
        control_layout.addWidget(self.normal_count_label)
        
        self.normal_target_input = QSpinBox()
        self.normal_target_input.setRange(0, 100000)
        self.normal_target_input.setValue(0)
        self.normal_target_input.setPrefix("목표: ")
        self.normal_target_input.setSuffix("개")
        self.normal_target_input.setMinimumWidth(120)
        control_layout.addWidget(self.normal_target_input)
        
        self.normal_delete_btn = QPushButton("Normal 랜덤 삭제")
        self.normal_delete_btn.clicked.connect(lambda: self.random_delete_images("normal"))
        self.normal_delete_btn.setEnabled(False)
        self.normal_delete_btn.setToolTip("목표 개수에 맞춰 랜덤하게 이미지 삭제")
        self.normal_delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #b8860b;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #daa520;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.normal_delete_btn)
        
        control_layout.addSpacing(20)
        
        # Defect Layer 이미지 개수 정보
        control_layout.addWidget(QLabel("Defect:"))
        self.defect_count_label = QLabel("0개")
        self.defect_count_label.setStyleSheet("QLabel { font-weight: bold; color: #ff6b6b; }")
        control_layout.addWidget(self.defect_count_label)
        
        self.defect_target_input = QSpinBox()
        self.defect_target_input.setRange(0, 100000)
        self.defect_target_input.setValue(0)
        self.defect_target_input.setPrefix("목표: ")
        self.defect_target_input.setSuffix("개")
        self.defect_target_input.setMinimumWidth(120)
        control_layout.addWidget(self.defect_target_input)
        
        self.defect_delete_btn = QPushButton("Defect 랜덤 삭제")
        self.defect_delete_btn.clicked.connect(lambda: self.random_delete_images("defect"))
        self.defect_delete_btn.setEnabled(False)
        self.defect_delete_btn.setToolTip("목표 개수에 맞춰 랜덤하게 이미지 삭제")
        self.defect_delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #b8860b;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #daa520;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.defect_delete_btn)
        
        control_layout.addStretch()
        
        # Normal 작업 버튼 3개 (Defect로 이동, 다음, 삭제)
        control_layout.addWidget(QLabel("[Normal]"))
        
        self.move_to_defect_btn = QPushButton("Defect로 이동")
        self.move_to_defect_btn.clicked.connect(self.move_image_to_defect)
        self.move_to_defect_btn.setEnabled(False)
        self.move_to_defect_btn.setToolTip("선택한 Normal 이미지를 Defect Layer로 이동")
        self.move_to_defect_btn.setStyleSheet("""
            QPushButton {
                background-color: #b22222;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #d22222;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.move_to_defect_btn)
        
        self.normal_next_btn = QPushButton("다음")
        self.normal_next_btn.clicked.connect(lambda: self.navigate_image("normal", 1))
        self.normal_next_btn.setEnabled(False)
        self.normal_next_btn.setToolTip("다음 이미지 (Space)")
        self.normal_next_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.normal_next_btn)
        
        self.reclassify_normal_btn = QPushButton("재분류")
        self.reclassify_normal_btn.clicked.connect(lambda: self.reclassify_image("normal"))
        self.reclassify_normal_btn.setEnabled(False)
        self.reclassify_normal_btn.setToolTip("재검토가 필요한 Normal 이미지를 재분류 폴더로 복사")
        self.reclassify_normal_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff8c00;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffa500;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.reclassify_normal_btn)
        
        self.delete_normal_btn = QPushButton("삭제")
        self.delete_normal_btn.clicked.connect(lambda: self.delete_selected_image("normal"))
        self.delete_normal_btn.setEnabled(False)
        self.delete_normal_btn.setToolTip("선택한 Normal 이미지를 완전히 삭제합니다")
        self.delete_normal_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a00000;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.delete_normal_btn)
        
        control_layout.addSpacing(30)
        
        # Defect 작업 버튼 3개 (Normal로 이동, 다음, 삭제)
        control_layout.addWidget(QLabel("[Defect]"))
        
        self.move_to_normal_btn = QPushButton("Normal로 이동")
        self.move_to_normal_btn.clicked.connect(self.move_image_to_normal)
        self.move_to_normal_btn.setEnabled(False)
        self.move_to_normal_btn.setToolTip("선택한 Defect 이미지를 Normal Layer로 이동")
        self.move_to_normal_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a7a1a;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #228a22;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.move_to_normal_btn)
        
        self.defect_next_btn = QPushButton("다음")
        self.defect_next_btn.clicked.connect(lambda: self.navigate_image("defect", 1))
        self.defect_next_btn.setEnabled(False)
        self.defect_next_btn.setToolTip("다음 이미지 (Space)")
        self.defect_next_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.defect_next_btn)
        
        self.reclassify_defect_btn = QPushButton("재분류")
        self.reclassify_defect_btn.clicked.connect(lambda: self.reclassify_image("defect"))
        self.reclassify_defect_btn.setEnabled(False)
        self.reclassify_defect_btn.setToolTip("재검토가 필요한 Defect 이미지를 재분류 폴더로 복사")
        self.reclassify_defect_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff8c00;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffa500;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.reclassify_defect_btn)
        
        self.delete_defect_btn = QPushButton("삭제")
        self.delete_defect_btn.clicked.connect(lambda: self.delete_selected_image("defect"))
        self.delete_defect_btn.setEnabled(False)
        self.delete_defect_btn.setToolTip("선택한 Defect 이미지를 완전히 삭제합니다")
        self.delete_defect_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a00000;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        control_layout.addWidget(self.delete_defect_btn)
        
        viewer_layout.addLayout(control_layout)
        
        # 듀얼 이미지 뷰어 (Normal / Defect) - 좌우 배치
        dual_viewer_splitter = QSplitter(Qt.Horizontal)
        
        # Normal Layer 뷰어 (왼쪽)
        normal_panel = QSplitter(Qt.Horizontal)
        normal_list_widget = QWidget()
        normal_list_layout = QVBoxLayout()
        self.normal_title_label = QLabel("Normal Layer 이미지:")
        self.normal_title_label.setStyleSheet("QLabel { font-weight: bold; }")
        normal_list_layout.addWidget(self.normal_title_label)
        self.normal_image_list = QListWidget()
        self.normal_image_list.itemClicked.connect(lambda: self.on_dual_image_selected("normal"))
        normal_list_layout.addWidget(self.normal_image_list)
        normal_list_widget.setLayout(normal_list_layout)
        
        self.normal_viewer = ImageViewer("Normal Layer")
        normal_panel.addWidget(normal_list_widget)
        normal_panel.addWidget(self.normal_viewer)
        normal_panel.setStretchFactor(0, 1)
        normal_panel.setStretchFactor(1, 2)
        
        # Defect Layer 뷰어 (오른쪽)
        defect_panel = QSplitter(Qt.Horizontal)
        defect_list_widget = QWidget()
        defect_list_layout = QVBoxLayout()
        self.defect_title_label = QLabel("Defect Layer 이미지:")
        self.defect_title_label.setStyleSheet("QLabel { font-weight: bold; }")
        defect_list_layout.addWidget(self.defect_title_label)
        self.defect_image_list = QListWidget()
        self.defect_image_list.itemClicked.connect(lambda: self.on_dual_image_selected("defect"))
        defect_list_layout.addWidget(self.defect_image_list)
        defect_list_widget.setLayout(defect_list_layout)
        
        self.defect_viewer = ImageViewer("Defect Layer")
        defect_panel.addWidget(defect_list_widget)
        defect_panel.addWidget(self.defect_viewer)
        defect_panel.setStretchFactor(0, 1)
        defect_panel.setStretchFactor(1, 2)
        
        # 좌우 분할
        dual_viewer_splitter.addWidget(normal_panel)
        dual_viewer_splitter.addWidget(defect_panel)
        
        viewer_layout.addWidget(dual_viewer_splitter, 1)
        viewer_tab.setLayout(viewer_layout)
        
        # 탭 추가
        tab_widget.addTab(log_tab, "실행 로그")
        tab_widget.addTab(viewer_tab, "듀얼 이미지 뷰어")
        
        return tab_widget
    
    def browse_root(self):
        """루트 디렉토리 선택"""
        folder = QFileDialog.getExistingDirectory(self, "루트 디렉토리 선택", self.root_input.text())
        if folder:
            self.root_input.setText(folder)
    
    def browse_layers(self):
        """레이어 파일 선택"""
        file, _ = QFileDialog.getOpenFileName(self, "레이어 파일 선택", "", "JSON Files (*.json)")
        if file:
            self.layers_input.setText(file)
    
    def browse_output(self):
        """출력 디렉토리 선택"""
        folder = QFileDialog.getExistingDirectory(self, "출력 디렉토리 선택", self.out_input.text())
        if folder:
            self.out_input.setText(folder)
    
    def start_sampling(self):
        """샘플링 시작"""
        self.log_text.clear()
        self.log_text.append("=" * 60)
        self.log_text.append("샘플링 시작...")
        self.log_text.append("=" * 60 + "\n")
        
        # 입력값 검증
        root = self.root_input.text()
        out = self.out_input.text()
        layers_file = self.layers_input.text()
        
        if not root or not out or not layers_file:
            QMessageBox.warning(self, "입력 오류", "모든 필드를 입력해주세요.")
            return
        
        # 키워드 파싱
        keywords = [k.strip() for k in self.keywords_input.text().split(',')]
        sample_ratio = self.ratio_input.value() / 100.0
        
        # UI 비활성화
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 무한 진행
        
        # 스레드 시작 (시드는 자동 생성)
        import time
        seed = int(time.time())  # 현재 시간을 시드로 사용
        
        # 샘플링 히스토리 로드
        sampling_history = self.load_sampling_history()
        
        self.sampler_thread = SamplerThread(
            root, out, layers_file, sample_ratio, seed, keywords,
            sampling_history=sampling_history
        )
        self.sampler_thread.progress.connect(self.update_log)
        self.sampler_thread.finished.connect(self.on_sampling_finished)
        self.sampler_thread.error.connect(self.on_sampling_error)
        self.sampler_thread.start()
    
    def update_log(self, message):
        """로그 업데이트"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def on_sampling_finished(self, result):
        """샘플링 완료"""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        stats = result["stats"]
        num_layers = result["num_layers"]
        self.current_output_dir = Path(result["output_dir"])
        
        # 샘플링 히스토리 저장
        if "sampling_history" in result:
            self.save_sampling_history(result["sampling_history"])
            self.log_text.append(f"\n[INFO] 샘플링 히스토리 업데이트됨 ({len(result['sampling_history'])}개 폴더)")
        
        self.log_text.append("\n" + "=" * 60)
        self.log_text.append("샘플링 완료!")
        self.log_text.append("=" * 60)
        
        # 통계 출력
        self.log_text.append("\n정상 이미지 (Normal Layer):")
        total_normal = 0
        for layer_id in range(1, num_layers + 1):
            count = stats["normal"].get(layer_id, 0)
            if count > 0:
                self.log_text.append(f"  Layer {layer_id:02d}: {count}개")
                total_normal += count
        self.log_text.append(f"  합계: {total_normal}개")
        
        self.log_text.append("\n불량 이미지 (Defect Layer):")
        total_defect = 0
        for layer_id in range(1, num_layers + 1):
            count = stats["defect"].get(layer_id, 0)
            if count > 0:
                self.log_text.append(f"  Layer {layer_id:02d}: {count}개")
                total_defect += count
        self.log_text.append(f"  합계: {total_defect}개")
        
        self.log_text.append(f"\n출력 폴더: {result['output_dir']}")
        self.log_text.append(f"총 {total_normal + total_defect}개 이미지 복사 완료")
        
        # 첫 번째 레이어 이미지 자동 로드
        self.load_layer_images(1)
        
        QMessageBox.information(
            self,
            "완료",
            f"샘플링이 완료되었습니다!\n\n"
            f"정상 이미지: {total_normal}개\n"
            f"불량 이미지: {total_defect}개\n"
            f"총 {total_normal + total_defect}개"
        )
    
    def on_sampling_error(self, error_msg):
        """샘플링 오류"""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log_text.append(f"\n오류 발생: {error_msg}")
        QMessageBox.critical(self, "오류", f"샘플링 중 오류가 발생했습니다:\n{error_msg}")
    
    def on_layer_changed(self, layer_id):
        """레이어 선택 변경"""
        self.load_layer_images(layer_id)
    
    def load_layer_images(self, layer_id, focus_category=None, focus_index=None, restore_position=False):
        """선택한 레이어의 이미지 로드
        
        Args:
            layer_id: 로드할 레이어 번호
            focus_category: 포커스를 유지할 카테고리 ("normal" 또는 "defect")
            focus_index: 포커스를 유지할 인덱스 (이동/삭제 후 다음 이미지로)
            restore_position: True면 마지막 저장된 위치로 복원
        """
        if not self.current_output_dir or not self.current_output_dir.exists():
            return
        
        # Normal Layer 이미지 로드
        self.normal_image_list.clear()
        normal_dir = self.current_output_dir / "Normal Layer" / f"Layer_{layer_id:02d}"
        if normal_dir.exists():
            # 파일 생성 시간 기준으로 정렬 (기존 이미지 먼저, 새 이미지 나중에)
            images = [f for f in normal_dir.iterdir() if f.suffix.lower() in ls.IMG_EXTS]
            images = sorted(images, key=lambda x: x.stat().st_ctime)
            self.normal_images[layer_id] = images
            for img in images:
                self.normal_image_list.addItem(img.name)
        else:
            self.normal_images[layer_id] = []
        
        # Defect Layer 이미지 로드
        self.defect_image_list.clear()
        defect_dir = self.current_output_dir / "Defect Layer" / f"Layer_{layer_id:02d}"
        if defect_dir.exists():
            # 파일 생성 시간 기준으로 정렬 (기존 이미지 먼저, 새 이미지 나중에)
            images = [f for f in defect_dir.iterdir() if f.suffix.lower() in ls.IMG_EXTS]
            images = sorted(images, key=lambda x: x.stat().st_ctime)
            self.defect_images[layer_id] = images
            for img in images:
                self.defect_image_list.addItem(img.name)
        else:
            self.defect_images[layer_id] = []
        
        # 이미지 개수 업데이트
        self.update_image_counts()
        
        # 포커스 복원
        if focus_category == "normal":
            normal_count = len(self.normal_images.get(layer_id, []))
            if normal_count > 0:
                # 다음 이미지로 (마지막이었다면 마지막-1)
                next_index = min(focus_index if focus_index is not None else 0, normal_count - 1)
                self.normal_image_list.setCurrentRow(next_index)
                self.normal_image_list.scrollToItem(
                    self.normal_image_list.currentItem(), 
                    QListWidget.PositionAtCenter
                )
                self.normal_viewer.display_image(self.normal_images[layer_id][next_index])
                self.move_to_defect_btn.setEnabled(True)
                self.delete_normal_btn.setEnabled(True)
                self.move_to_normal_btn.setEnabled(False)
                self.delete_defect_btn.setEnabled(False)
                self.last_focused_category = "normal"
                # 타이틀 업데이트
                self.update_title_labels()
            else:
                self.normal_viewer.clear()
                self.move_to_defect_btn.setEnabled(False)
                self.delete_normal_btn.setEnabled(False)
                
        elif focus_category == "defect":
            defect_count = len(self.defect_images.get(layer_id, []))
            if defect_count > 0:
                # 다음 이미지로 (마지막이었다면 마지막-1)
                next_index = min(focus_index if focus_index is not None else 0, defect_count - 1)
                self.defect_image_list.setCurrentRow(next_index)
                self.defect_image_list.scrollToItem(
                    self.defect_image_list.currentItem(), 
                    QListWidget.PositionAtCenter
                )
                self.defect_viewer.display_image(self.defect_images[layer_id][next_index])
                self.move_to_normal_btn.setEnabled(True)
                self.delete_defect_btn.setEnabled(True)
                self.move_to_defect_btn.setEnabled(False)
                self.delete_normal_btn.setEnabled(False)
                self.last_focused_category = "defect"
                # 타이틀 업데이트
                self.update_title_labels()
            else:
                self.defect_viewer.clear()
                self.move_to_normal_btn.setEnabled(False)
                self.delete_defect_btn.setEnabled(False)
        elif restore_position:
            # 마지막 저장된 위치 복원
            last_pos = self.load_last_position()
            if last_pos and last_pos.get("layer") == layer_id:
                restore_cat = last_pos.get("category")
                restore_idx = last_pos.get("index", 0)
                
                if restore_cat == "normal" and restore_idx < len(self.normal_images.get(layer_id, [])):
                    self.normal_image_list.setCurrentRow(restore_idx)
                    self.normal_image_list.scrollToItem(
                        self.normal_image_list.currentItem(), 
                        QListWidget.PositionAtCenter
                    )
                    self.normal_viewer.display_image(self.normal_images[layer_id][restore_idx])
                    self.move_to_defect_btn.setEnabled(True)
                    self.normal_next_btn.setEnabled(True)
                    self.reclassify_normal_btn.setEnabled(True)
                    self.delete_normal_btn.setEnabled(True)
                    self.move_to_normal_btn.setEnabled(False)
                    self.defect_next_btn.setEnabled(False)
                    self.reclassify_defect_btn.setEnabled(False)
                    self.delete_defect_btn.setEnabled(False)
                    self.last_focused_category = "normal"
                    self.update_title_labels()
                elif restore_cat == "defect" and restore_idx < len(self.defect_images.get(layer_id, [])):
                    self.defect_image_list.setCurrentRow(restore_idx)
                    self.defect_image_list.scrollToItem(
                        self.defect_image_list.currentItem(), 
                        QListWidget.PositionAtCenter
                    )
                    self.defect_viewer.display_image(self.defect_images[layer_id][restore_idx])
                    self.move_to_normal_btn.setEnabled(True)
                    self.defect_next_btn.setEnabled(True)
                    self.reclassify_defect_btn.setEnabled(True)
                    self.delete_defect_btn.setEnabled(True)
                    self.move_to_defect_btn.setEnabled(False)
                    self.normal_next_btn.setEnabled(False)
                    self.reclassify_normal_btn.setEnabled(False)
                    self.delete_normal_btn.setEnabled(False)
                    self.last_focused_category = "defect"
                    self.update_title_labels()
        else:
            # 포커스 지정 없으면 뷰어만 초기화
            self.normal_viewer.clear()
            self.defect_viewer.clear()
    
    def on_dual_image_selected(self, category):
        """듀얼 뷰어에서 이미지 선택"""
        layer_id = self.layer_selector.value()
        
        if category == "normal":
            index = self.normal_image_list.currentRow()
            images = self.normal_images.get(layer_id, [])
            if 0 <= index < len(images):
                # 선택된 항목이 보이도록 스크롤
                self.normal_image_list.scrollToItem(
                    self.normal_image_list.currentItem(), 
                    QListWidget.PositionAtCenter
                )
                self.normal_viewer.display_image(images[index])
                # Normal 선택 시 Normal 버튼 활성화, Defect 버튼 비활성화
                self.move_to_defect_btn.setEnabled(True)
                self.normal_next_btn.setEnabled(True)
                self.reclassify_normal_btn.setEnabled(True)
                self.delete_normal_btn.setEnabled(True)
                self.move_to_normal_btn.setEnabled(False)
                self.defect_next_btn.setEnabled(False)
                self.reclassify_defect_btn.setEnabled(False)
                self.delete_defect_btn.setEnabled(False)
                self.last_focused_category = "normal"
                # 마지막 위치 저장
                self.save_last_position(layer_id, "normal", index)
                # 타이틀 업데이트
                self.update_title_labels()
        else:
            index = self.defect_image_list.currentRow()
            images = self.defect_images.get(layer_id, [])
            if 0 <= index < len(images):
                # 선택된 항목이 보이도록 스크롤
                self.defect_image_list.scrollToItem(
                    self.defect_image_list.currentItem(), 
                    QListWidget.PositionAtCenter
                )
                self.defect_viewer.display_image(images[index])
                # Defect 선택 시 Defect 버튼 활성화, Normal 버튼 비활성화
                self.move_to_normal_btn.setEnabled(True)
                self.defect_next_btn.setEnabled(True)
                self.reclassify_defect_btn.setEnabled(True)
                self.delete_defect_btn.setEnabled(True)
                self.move_to_defect_btn.setEnabled(False)
                self.normal_next_btn.setEnabled(False)
                self.reclassify_normal_btn.setEnabled(False)
                self.delete_normal_btn.setEnabled(False)
                self.last_focused_category = "defect"
                # 마지막 위치 저장
                self.save_last_position(layer_id, "defect", index)
                # 타이틀 업데이트
                self.update_title_labels()
    
    def open_image_viewer(self):
        """출력 폴더의 이미지를 바로 뷰어로 열기"""
        out_dir = Path(self.out_input.text())
        
        if not out_dir.exists():
            QMessageBox.warning(
                self, 
                "경고", 
                f"출력 디렉토리가 존재하지 않습니다.\n\n{out_dir}\n\n"
                "먼저 샘플링을 실행하거나 기존 출력 폴더를 선택해주세요."
            )
            return
        
        # Normal Layer, Defect Layer 폴더 확인
        normal_dir = out_dir / "Normal Layer"
        defect_dir = out_dir / "Defect Layer"
        
        if not normal_dir.exists() and not defect_dir.exists():
            QMessageBox.warning(
                self,
                "경고",
                f"Normal Layer 또는 Defect Layer 폴더를 찾을 수 없습니다.\n\n"
                f"출력 디렉토리: {out_dir}\n\n"
                "먼저 샘플링을 실행해주세요."
            )
            return
        
        # 출력 디렉토리 설정
        self.current_output_dir = out_dir
        
        # 마지막 위치 복원 또는 첫 번째 레이어 로드
        last_pos = self.load_last_position()
        if last_pos and last_pos.get("layer"):
            layer_to_load = last_pos["layer"]
            self.layer_selector.setValue(layer_to_load)
            self.load_layer_images(layer_to_load, restore_position=True)
        else:
            self.load_layer_images(1)
        
        # 이미지 뷰어 탭으로 전환
        tab_widget = self.centralWidget().findChild(QTabWidget)
        if tab_widget:
            tab_widget.setCurrentIndex(1)  # 두 번째 탭 (이미지 뷰어)
        
        self.log_text.append(f"\n이미지 뷰어 열림: {out_dir}")
        
        # 통계 표시
        total_normal = 0
        total_defect = 0
        
        if normal_dir.exists():
            for layer_folder in normal_dir.iterdir():
                if layer_folder.is_dir():
                    count = len([f for f in layer_folder.iterdir() if f.suffix.lower() in ls.IMG_EXTS])
                    total_normal += count
        
        if defect_dir.exists():
            for layer_folder in defect_dir.iterdir():
                if layer_folder.is_dir():
                    count = len([f for f in layer_folder.iterdir() if f.suffix.lower() in ls.IMG_EXTS])
                    total_defect += count
        
        self.log_text.append(f"Normal Layer: {total_normal}개 이미지")
        self.log_text.append(f"Defect Layer: {total_defect}개 이미지")
        
        QMessageBox.information(
            self,
            "이미지 뷰어",
            f"이미지 뷰어를 열었습니다.\n\n"
            f"Normal Layer: {total_normal}개\n"
            f"Defect Layer: {total_defect}개\n\n"
            f"레이어를 선택하여 이미지를 확인하세요."
        )
    
    def move_image_to_normal(self):
        """선택한 Defect 이미지를 Normal Layer로 이동"""
        if not self.current_output_dir:
            return
        
        layer_id = self.layer_selector.value()
        defect_index = self.defect_image_list.currentRow()
        
        if defect_index < 0 or layer_id not in self.defect_images:
            QMessageBox.warning(self, "경고", "이동할 이미지를 선택해주세요.")
            return
        
        images = self.defect_images[layer_id]
        if defect_index >= len(images):
            return
        
        src_path = images[defect_index]
        
        # 대상 폴더 생성
        normal_layer_dir = self.current_output_dir / "Normal Layer" / f"Layer_{layer_id:02d}"
        ls.ensure_dir(normal_layer_dir)
        
        # 파일 이동
        dst_path = normal_layer_dir / src_path.name
        try:
            shutil.move(str(src_path), str(dst_path))
            
            # Undo 스택에 추가 (현재 인덱스 저장)
            self.add_undo_action("move", {
                "old_path": src_path,
                "new_path": dst_path,
                "from_category": "Defect",
                "to_category": "Normal",
                "layer": layer_id,
                "filename": src_path.name,
                "defect_index": defect_index  # 현재 위치 저장
            })
            
            self.log_text.append(f"[이동] Defect → Normal | Layer {layer_id:02d}: {src_path.name}")
            
            # 리스트 갱신 (현재 카테고리 위치 유지)
            self.load_layer_images(layer_id, focus_category="defect", focus_index=defect_index)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 이동 중 오류 발생:\n{str(e)}")
    
    def move_image_to_defect(self):
        """선택한 Normal 이미지를 Defect Layer로 이동"""
        if not self.current_output_dir:
            return
        
        layer_id = self.layer_selector.value()
        normal_index = self.normal_image_list.currentRow()
        
        if normal_index < 0 or layer_id not in self.normal_images:
            QMessageBox.warning(self, "경고", "이동할 이미지를 선택해주세요.")
            return
        
        images = self.normal_images[layer_id]
        if normal_index >= len(images):
            return
        
        src_path = images[normal_index]
        
        # 대상 폴더 생성
        defect_layer_dir = self.current_output_dir / "Defect Layer" / f"Layer_{layer_id:02d}"
        ls.ensure_dir(defect_layer_dir)
        
        # 파일 이동
        dst_path = defect_layer_dir / src_path.name
        try:
            shutil.move(str(src_path), str(dst_path))
            
            # Undo 스택에 추가 (현재 인덱스 저장)
            self.add_undo_action("move", {
                "old_path": src_path,
                "new_path": dst_path,
                "from_category": "Normal",
                "to_category": "Defect",
                "layer": layer_id,
                "filename": src_path.name,
                "normal_index": normal_index  # 현재 위치 저장
            })
            
            self.log_text.append(f"[이동] Normal → Defect | Layer {layer_id:02d}: {src_path.name}")
            
            # 리스트 갱신 (현재 카테고리 위치 유지)
            self.load_layer_images(layer_id, focus_category="normal", focus_index=normal_index)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 이동 중 오류 발생:\n{str(e)}")
    
    def reclassify_image(self, category):
        """선택한 이미지를 재분류 폴더로 복사
        
        Args:
            category: "normal" 또는 "defect"
        """
        if not self.current_output_dir:
            return
        
        layer_id = self.layer_selector.value()
        
        if category == "normal":
            index = self.normal_image_list.currentRow()
            images = self.normal_images.get(layer_id, [])
            category_name = "Normal"
        else:
            index = self.defect_image_list.currentRow()
            images = self.defect_images.get(layer_id, [])
            category_name = "Defect"
        
        if index < 0 or not images or index >= len(images):
            QMessageBox.warning(self, "경고", "재분류할 이미지를 선택해주세요.")
            return
        
        img_path = images[index]
        
        # 재분류 폴더 생성
        reclassify_base = self.current_output_dir / "Reclassify"
        if category == "normal":
            reclassify_dir = reclassify_base / "Normal Layer" / f"Layer_{layer_id:02d}"
        else:
            reclassify_dir = reclassify_base / "Defect Layer" / f"Layer_{layer_id:02d}"
        
        ls.ensure_dir(reclassify_dir)
        
        # 이미지 이동 (원본 폴더에서 제거)
        dst_path = reclassify_dir / img_path.name
        
        try:
            # 이미 이동된 파일이 있는지 확인
            if dst_path.exists():
                # 기존 파일 삭제 (조용히 덮어쓰기)
                dst_path.unlink()
            
            # 파일 이동
            shutil.move(str(img_path), str(dst_path))
            
            # Undo 스택에 추가
            self.add_undo_action("reclassify", {
                "old_path": img_path,
                "new_path": dst_path,
                "category": category_name,
                "layer": layer_id,
                "filename": img_path.name,
                "index": index
            })
            
            self.log_text.append(f"[재분류] {category_name} Layer {layer_id:02d}: {img_path.name} → Reclassify 폴더로 이동")
            
            # 리스트 갱신 (현재 위치 유지)
            self.load_layer_images(layer_id, focus_category=category, focus_index=index)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 이동 중 오류 발생:\n{str(e)}")
    
    def delete_selected_image(self, category=None):
        """선택한 이미지 삭제
        
        Args:
            category: "normal" 또는 "defect", None이면 last_focused_category 사용
        """
        if not self.current_output_dir:
            return
        
        layer_id = self.layer_selector.value()
        if category is None:
            category = self.last_focused_category
        
        if category == "normal":
            index = self.normal_image_list.currentRow()
            images = self.normal_images.get(layer_id, [])
            category_name = "Normal"
        else:
            index = self.defect_image_list.currentRow()
            images = self.defect_images.get(layer_id, [])
            category_name = "Defect"
        
        if index < 0 or not images or index >= len(images):
            QMessageBox.warning(self, "경고", "삭제할 이미지를 선택해주세요.")
            return
        
        img_path = images[index]
        
        # 확인 대화상자
        reply = QMessageBox.question(
            self,
            "이미지 삭제 확인",
            f"{category_name} Layer {layer_id:02d}\n\n"
            f"파일: {img_path.name}\n\n"
            f"이 이미지를 완전히 삭제하시겠습니까?\n"
            "(삭제된 이미지는 복구할 수 없습니다)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # 이미지 삭제
        try:
            # Undo 스택에 추가 (삭제는 복구 불가하지만 기록은 남김)
            self.add_undo_action("delete", {
                "path": img_path,
                "category": category_name,
                "layer": layer_id,
                "filename": img_path.name
            })
            
            img_path.unlink()  # 파일 삭제
            self.log_text.append(f"[삭제] {category_name} Layer {layer_id:02d}: {img_path.name}")
            
            # 리스트 갱신 (현재 위치 유지)
            focus_category = category
            self.load_layer_images(layer_id, focus_category=focus_category, focus_index=index)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 삭제 중 오류 발생:\n{str(e)}")
    
    def setup_shortcuts(self):
        """키보드 단축키 설정"""
        # Space: 다음 이미지
        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(self.on_space_pressed)
        
        # Backspace: 이전 이미지
        backspace_shortcut = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        backspace_shortcut.activated.connect(self.on_backspace_pressed)
        
        # Delete: 이미지 삭제
        delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        delete_shortcut.activated.connect(self.delete_selected_image)
        
        # Ctrl+Z: 실행 취소
        undo_shortcut = QShortcut(QKeySequence.Undo, self)  # Ctrl+Z
        undo_shortcut.activated.connect(self.undo_last_action)
    
    def on_space_pressed(self):
        """Space 키: 다음 이미지"""
        self.navigate_image(self.last_focused_category, 1)
    
    def on_backspace_pressed(self):
        """Backspace 키: 이전 이미지"""
        self.navigate_image(self.last_focused_category, -1)
    
    def add_undo_action(self, action_type, data):
        """실행 취소 스택에 작업 추가
        
        Args:
            action_type: "move" 또는 "delete"
            data: 작업 정보 딕셔너리
        """
        self.undo_stack.append({
            "type": action_type,
            "data": data
        })
        
        # 최대 개수 제한
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        
        # UI 업데이트
        self.update_undo_label()
    
    def undo_last_action(self):
        """마지막 작업 실행 취소 (Ctrl+Z)"""
        if not self.undo_stack:
            self.log_text.append("[Undo] 되돌릴 작업이 없습니다.")
            return
        
        if not self.current_output_dir:
            return
        
        action = self.undo_stack.pop()
        action_type = action["type"]
        data = action["data"]
        
        try:
            if action_type == "move":
                # 이미지 이동 되돌리기
                src_path = data["new_path"]
                dst_path = data["old_path"]
                
                if not src_path.exists():
                    self.log_text.append(f"[Undo 실패] 파일을 찾을 수 없음: {src_path.name}")
                    return
                
                shutil.move(str(src_path), str(dst_path))
                
                self.log_text.append(
                    f"[Undo] {data['from_category']} ← {data['to_category']} | "
                    f"Layer {data['layer']:02d}: {data['filename']}"
                )
                
                # 리스트 갱신 (원래 위치로 복원)
                if data['from_category'] == "Normal":
                    # Normal에서 Defect로 이동했던 것을 되돌림 → Normal 위치 복원
                    focus_category = "normal"
                    focus_index = data.get("normal_index", 0)
                else:
                    # Defect에서 Normal로 이동했던 것을 되돌림 → Defect 위치 복원
                    focus_category = "defect"
                    focus_index = data.get("defect_index", 0)
                
                self.load_layer_images(data['layer'], focus_category=focus_category, focus_index=focus_index)
                
            elif action_type == "reclassify":
                # 재분류 되돌리기 (Reclassify → 원래 폴더)
                src_path = data["new_path"]
                dst_path = data["old_path"]
                
                if not src_path.exists():
                    self.log_text.append(f"[Undo 실패] 파일을 찾을 수 없음: {src_path.name}")
                    return
                
                shutil.move(str(src_path), str(dst_path))
                
                self.log_text.append(
                    f"[Undo] 재분류 취소 | "
                    f"{data['category']} Layer {data['layer']:02d}: {data['filename']}"
                )
                
                # 리스트 갱신 (원래 위치로 복원)
                focus_category = data['category'].lower()
                focus_index = data.get("index", 0)
                self.load_layer_images(data['layer'], focus_category=focus_category, focus_index=focus_index)
                
            elif action_type == "delete":
                # 삭제는 복구 불가능하므로 경고만 표시
                self.log_text.append(
                    f"[Undo 불가] 삭제된 이미지는 복구할 수 없습니다: {data['filename']}"
                )
                QMessageBox.warning(
                    self,
                    "실행 취소 불가",
                    f"삭제된 이미지는 복구할 수 없습니다.\n\n"
                    f"파일: {data['filename']}\n"
                    f"카테고리: {data['category']}\n"
                    f"레이어: {data['layer']}"
                )
                return
            
            # UI 업데이트
            self.update_undo_label()
            
        except Exception as e:
            self.log_text.append(f"[Undo 오류] {str(e)}")
            QMessageBox.critical(self, "오류", f"실행 취소 중 오류 발생:\n{str(e)}")
    
    def update_undo_label(self):
        """Undo 레이블 업데이트"""
        count = len(self.undo_stack)
        if count > 0:
            # 마지막 작업 정보
            last_action = self.undo_stack[-1]
            action_type = last_action["type"]
            action_name = "이동" if action_type == "move" else ("재분류" if action_type == "reclassify" else "삭제")
            self.undo_label.setText(f"실행 취소 가능: {count}개 (Ctrl+Z) - 마지막: {action_name}")
            self.undo_label.setStyleSheet("QLabel { color: #4a9eff; font-size: 9pt; padding: 5px; }")
        else:
            self.undo_label.setText("실행 취소 가능: 0개 (Ctrl+Z)")
            self.undo_label.setStyleSheet("QLabel { color: #888; font-size: 9pt; padding: 5px; }")
    
    def navigate_image(self, category, direction):
        """이미지 네비게이션
        
        Args:
            category: "normal" 또는 "defect"
            direction: 1 (다음) 또는 -1 (이전)
        """
        layer_id = self.layer_selector.value()
        
        if category == "normal":
            image_list = self.normal_image_list
            images = self.normal_images.get(layer_id, [])
            viewer = self.normal_viewer
            self.last_focused_category = "normal"
        else:
            image_list = self.defect_image_list
            images = self.defect_images.get(layer_id, [])
            viewer = self.defect_viewer
            self.last_focused_category = "defect"
        
        if not images:
            return
        
        current_index = image_list.currentRow()
        
        # 첫 실행 시 또는 선택 없을 때
        if current_index < 0:
            new_index = 0 if direction > 0 else len(images) - 1
        else:
            new_index = current_index + direction
        
        # 순환 (마지막 → 처음, 처음 → 마지막)
        if new_index >= len(images):
            new_index = 0
        elif new_index < 0:
            new_index = len(images) - 1
        
        # 리스트 선택 및 뷰어 업데이트
        image_list.setCurrentRow(new_index)
        # 선택된 항목이 보이도록 스크롤
        image_list.scrollToItem(image_list.currentItem(), QListWidget.PositionAtCenter)
        viewer.display_image(images[new_index])
        
        # 이동/삭제 버튼 상태 업데이트
        if category == "normal":
            self.move_to_defect_btn.setEnabled(True)
            self.normal_next_btn.setEnabled(True)
            self.reclassify_normal_btn.setEnabled(True)
            self.delete_normal_btn.setEnabled(True)
            self.move_to_normal_btn.setEnabled(False)
            self.defect_next_btn.setEnabled(False)
            self.reclassify_defect_btn.setEnabled(False)
            self.delete_defect_btn.setEnabled(False)
        else:
            self.move_to_normal_btn.setEnabled(True)
            self.defect_next_btn.setEnabled(True)
            self.reclassify_defect_btn.setEnabled(True)
            self.delete_defect_btn.setEnabled(True)
            self.move_to_defect_btn.setEnabled(False)
            self.normal_next_btn.setEnabled(False)
            self.reclassify_normal_btn.setEnabled(False)
            self.delete_normal_btn.setEnabled(False)
        
        # 타이틀 업데이트
        self.update_title_labels()
    
    def update_image_counts(self):
        """현재 레이어의 이미지 개수 업데이트"""
        layer_id = self.layer_selector.value()
        
        # Normal 개수
        normal_count = len(self.normal_images.get(layer_id, []))
        self.normal_count_label.setText(f"{normal_count}개")
        
        # Defect 개수
        defect_count = len(self.defect_images.get(layer_id, []))
        self.defect_count_label.setText(f"{defect_count}개")
        
        # 삭제 버튼 활성화 (이미지가 있을 때만)
        self.normal_delete_btn.setEnabled(normal_count > 0)
        self.defect_delete_btn.setEnabled(defect_count > 0)
        
        # 타이틀 레이블 업데이트
        self.update_title_labels()
    
    def update_title_labels(self):
        """Normal/Defect 타이틀에 현재 위치/총합 표시"""
        layer_id = self.layer_selector.value()
        
        # Normal 타이틀
        normal_count = len(self.normal_images.get(layer_id, []))
        normal_index = self.normal_image_list.currentRow()
        if normal_count > 0:
            if normal_index >= 0:
                self.normal_title_label.setText(f"Normal Layer 이미지: [{normal_index + 1}/{normal_count}]")
            else:
                self.normal_title_label.setText(f"Normal Layer 이미지: [-/{normal_count}]")
        else:
            self.normal_title_label.setText(f"Normal Layer 이미지: [0/0]")
        
        # Defect 타이틀
        defect_count = len(self.defect_images.get(layer_id, []))
        defect_index = self.defect_image_list.currentRow()
        if defect_count > 0:
            if defect_index >= 0:
                self.defect_title_label.setText(f"Defect Layer 이미지: [{defect_index + 1}/{defect_count}]")
            else:
                self.defect_title_label.setText(f"Defect Layer 이미지: [-/{defect_count}]")
        else:
            self.defect_title_label.setText(f"Defect Layer 이미지: [0/0]")
    
    def random_delete_images(self, category):
        """랜덤하게 이미지 삭제하여 목표 개수에 맞춤
        
        Args:
            category: "normal" 또는 "defect"
        """
        if not self.current_output_dir:
            return
        
        layer_id = self.layer_selector.value()
        
        if category == "normal":
            images = self.normal_images.get(layer_id, [])
            target_count = self.normal_target_input.value()
            category_name = "Normal"
        else:
            images = self.defect_images.get(layer_id, [])
            target_count = self.defect_target_input.value()
            category_name = "Defect"
        
        current_count = len(images)
        
        if current_count == 0:
            QMessageBox.warning(self, "경고", f"{category_name} Layer에 이미지가 없습니다.")
            return
        
        if target_count >= current_count:
            QMessageBox.information(
                self,
                "안내",
                f"현재 이미지 개수({current_count}개)가 목표 개수({target_count}개) 이하입니다.\n"
                "삭제할 이미지가 없습니다."
            )
            return
        
        delete_count = current_count - target_count
        
        # 확인 대화상자
        reply = QMessageBox.question(
            self,
            "이미지 삭제 확인",
            f"{category_name} Layer {layer_id:02d}\n\n"
            f"현재 개수: {current_count}개\n"
            f"목표 개수: {target_count}개\n"
            f"삭제할 개수: {delete_count}개\n\n"
            f"랜덤하게 {delete_count}개의 이미지를 삭제하시겠습니까?\n"
            "(삭제된 이미지는 복구할 수 없습니다)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # 랜덤하게 삭제할 이미지 선택
        images_to_delete = random.sample(images, delete_count)
        
        # 이미지 삭제
        deleted_count = 0
        failed_files = []
        
        for img_path in images_to_delete:
            try:
                img_path.unlink()  # 파일 삭제
                deleted_count += 1
                self.log_text.append(f"[삭제] {category_name} Layer {layer_id:02d}: {img_path.name}")
            except Exception as e:
                failed_files.append(f"{img_path.name}: {str(e)}")
        
        # 리스트 갱신
        self.load_layer_images(layer_id)
        
        # 결과 메시지
        if failed_files:
            QMessageBox.warning(
                self,
                "삭제 완료 (일부 실패)",
                f"{category_name} Layer {layer_id:02d}\n\n"
                f"삭제 성공: {deleted_count}개\n"
                f"삭제 실패: {len(failed_files)}개\n\n"
                f"실패한 파일:\n" + "\n".join(failed_files[:5]) +
                ("\n..." if len(failed_files) > 5 else "")
            )
        else:
            QMessageBox.information(
                self,
                "삭제 완료",
                f"{category_name} Layer {layer_id:02d}\n\n"
                f"{deleted_count}개의 이미지가 삭제되었습니다.\n"
                f"현재 개수: {len(self.normal_images.get(layer_id, []) if category == 'normal' else self.defect_images.get(layer_id, []))}개"
            )


    def load_sampling_history(self):
        """샘플링 히스토리 로드"""
        if not self.sampling_history_file.exists():
            return {}
        
        try:
            with open(self.sampling_history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log_text.append(f"[경고] 히스토리 파일 로드 실패: {e}")
            return {}
    
    def save_sampling_history(self, history):
        """샘플링 히스토리 저장"""
        try:
            with open(self.sampling_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log_text.append(f"[오류] 히스토리 저장 실패: {e}")
    
    def save_last_position(self, layer_id, category, index):
        """마지막 선택 위치 저장"""
        try:
            position = {
                "layer": layer_id,
                "category": category,
                "index": index
            }
            with open(self.last_position_file, 'w', encoding='utf-8') as f:
                json.dump(position, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # 저장 실패는 조용히 무시
            pass
    
    def load_last_position(self):
        """마지막 선택 위치 로드"""
        if not self.last_position_file.exists():
            return None
        
        try:
            with open(self.last_position_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return None
    
    def clear_sampling_history(self):
        """샘플링 히스토리 초기화"""
        reply = QMessageBox.question(
            self,
            "리스트 클린 확인",
            "샘플링 히스토리를 모두 삭제하시겠습니까?\n\n"
            "이 작업은 복구할 수 없으며, 다음 샘플링 시\n"
            "이전에 선택된 이미지도 다시 선택될 수 있습니다.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if self.sampling_history_file.exists():
                    self.sampling_history_file.unlink()
                    self.log_text.append("[INFO] 샘플링 히스토리가 삭제되었습니다.")
                    QMessageBox.information(self, "완료", "샘플링 히스토리가 삭제되었습니다.")
                else:
                    QMessageBox.information(self, "안내", "삭제할 히스토리가 없습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"히스토리 삭제 중 오류 발생:\n{e}")
    
    def view_sampling_history(self):
        """샘플링 히스토리 보기"""
        history = self.load_sampling_history()
        
        if not history:
            QMessageBox.information(self, "히스토리", "샘플링 히스토리가 없습니다.")
            return
        
        # 통계 계산
        total_count = sum(len(images) for images in history.values())
        folder_count = len(history)
        
        msg = f"샘플링 히스토리 정보\n\n"
        msg += f"총 폴더 수: {folder_count}개\n"
        msg += f"총 이미지 수: {total_count}개\n\n"
        msg += "폴더별 이미지 수:\n"
        
        for folder_name, images in sorted(history.items())[:10]:  # 상위 10개만 표시
            msg += f"  {folder_name}: {len(images)}개\n"
        
        if folder_count > 10:
            msg += f"  ... 외 {folder_count - 10}개 폴더"
        
        QMessageBox.information(self, "샘플링 히스토리", msg)
    
    def open_partial_sampling_dialog(self):
        """부분 샘플링 다이얼로그 열기"""
        dialog = PartialSamplingDialog(self)
        if dialog.exec():
            # 다이얼로그에서 설정된 값으로 부분 샘플링 실행
            self.start_partial_sampling(dialog.get_settings())
    
    def start_partial_sampling(self, settings):
        """부분 샘플링 시작"""
        self.log_text.append("\n" + "=" * 60)
        self.log_text.append("부분 샘플링 시작...")
        self.log_text.append("=" * 60)
        
        layer = settings["layer"]
        category_idx = settings["category"]
        ratio = settings["ratio"]
        
        # 카테고리 이름 매핑
        category_names = {0: "Normal + Defect", 1: "Normal만", 2: "Defect만"}
        category_name = category_names[category_idx]
        
        # target_category 매핑: 0: both, 1: normal only, 2: defect only -> None, 0, 1
        target_category = None if category_idx == 0 else (0 if category_idx == 1 else 1)
        
        self.log_text.append(f"레이어: {'모든 레이어' if layer == 0 else f'Layer {layer:02d}'}")
        self.log_text.append(f"카테고리: {category_name}")
        self.log_text.append(f"샘플링 비율: {ratio * 100:.1f}%\n")
        
        # 입력값 검증
        root = self.root_input.text()
        out = self.out_input.text()
        layers_file = self.layers_input.text()
        
        if not root or not out or not layers_file:
            QMessageBox.warning(self, "입력 오류", "모든 필드를 입력해주세요.")
            return
        
        # 키워드 파싱
        keywords = [k.strip() for k in self.keywords_input.text().split(',')]
        
        # UI 비활성화
        self.run_btn.setEnabled(False)
        self.partial_sampling_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 무한 진행
        
        # 스레드 시작 (시드는 자동 생성)
        import time
        seed = int(time.time())  # 현재 시간을 시드로 사용
        
        # 샘플링 히스토리 로드
        sampling_history = self.load_sampling_history()
        
        self.sampler_thread = SamplerThread(
            root, out, layers_file, ratio, seed, keywords,
            sampling_history=sampling_history,
            target_layer=layer if layer != 0 else None,
            target_category=target_category
        )
        self.sampler_thread.progress.connect(self.update_log)
        self.sampler_thread.finished.connect(self.on_partial_sampling_finished)
        self.sampler_thread.error.connect(self.on_sampling_error)
        self.sampler_thread.start()
    
    def on_partial_sampling_finished(self, result):
        """부분 샘플링 완료"""
        self.run_btn.setEnabled(True)
        self.partial_sampling_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        stats = result["stats"]
        num_layers = result["num_layers"]
        self.current_output_dir = Path(result["output_dir"])
        
        # 샘플링 히스토리 저장
        if "sampling_history" in result:
            self.save_sampling_history(result["sampling_history"])
            self.log_text.append(f"\n[INFO] 샘플링 히스토리 업데이트됨 ({len(result['sampling_history'])}개 폴더)")
        
        self.log_text.append("\n" + "=" * 60)
        self.log_text.append("부분 샘플링 완료!")
        self.log_text.append("=" * 60)
        
        # 통계 출력
        total_normal = 0
        total_defect = 0
        
        self.log_text.append("\n추가된 이미지:")
        for layer_id in range(1, num_layers + 1):
            normal_count = stats["normal"].get(layer_id, 0)
            defect_count = stats["defect"].get(layer_id, 0)
            
            if normal_count > 0 or defect_count > 0:
                self.log_text.append(f"  Layer {layer_id:02d}: Normal {normal_count}개, Defect {defect_count}개")
                total_normal += normal_count
                total_defect += defect_count
        
        self.log_text.append(f"\n총 추가: Normal {total_normal}개, Defect {total_defect}개")
        self.log_text.append(f"출력 폴더: {result['output_dir']}")
        
        # 현재 보고 있는 레이어 새로고침 (포커스 유지)
        current_layer = self.layer_selector.value()
        
        # 현재 선택된 카테고리와 인덱스 저장
        if self.last_focused_category == "normal":
            current_index = self.normal_image_list.currentRow()
            focus_category = "normal"
        else:
            current_index = self.defect_image_list.currentRow()
            focus_category = "defect"
        
        # 리스트 갱신 (현재 위치 유지)
        if current_index >= 0:
            self.load_layer_images(current_layer, focus_category=focus_category, focus_index=current_index)
        else:
            self.load_layer_images(current_layer)
        
        QMessageBox.information(
            self,
            "완료",
            f"부분 샘플링이 완료되었습니다!\n\n"
            f"추가된 이미지:\n"
            f"  Normal: {total_normal}개\n"
            f"  Defect: {total_defect}개\n"
            f"총 {total_normal + total_defect}개\n\n"
            f"기존 작업 위치가 유지되었습니다."
        )


class PartialSamplingDialog(QWidget):
    """부분 샘플링 설정 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.init_ui()
    
    def init_ui(self):
        from PyQt5.QtWidgets import QDialog, QComboBox
        
        # QDialog 생성
        self.dialog = QDialog(self.parent_gui)
        self.dialog.setWindowTitle("부분 샘플링 설정")
        self.dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout()
        
        # 설명
        info_label = QLabel(
            "특정 레이어와 카테고리에서 추가 샘플링을 수행합니다.\n"
            "이미 샘플링된 이미지는 자동으로 제외됩니다."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { padding: 10px; background-color: #2a2a2a; border-radius: 5px; }")
        layout.addWidget(info_label)
        
        # 설정 그룹
        settings_group = QGroupBox("샘플링 설정")
        settings_layout = QGridLayout()
        
        # 레이어 선택
        settings_layout.addWidget(QLabel("레이어:"), 0, 0)
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("모든 레이어", 0)
        for i in range(1, 20):
            self.layer_combo.addItem(f"Layer {i:02d}", i)
        settings_layout.addWidget(self.layer_combo, 0, 1)
        
        # 카테고리 선택
        settings_layout.addWidget(QLabel("카테고리:"), 1, 0)
        self.category_combo = QComboBox()
        self.category_combo.addItems(["Normal + Defect", "Normal만", "Defect만"])
        settings_layout.addWidget(self.category_combo, 1, 1)
        
        # 샘플링 비율
        settings_layout.addWidget(QLabel("샘플링 비율:"), 2, 0)
        self.ratio_input = QDoubleSpinBox()
        self.ratio_input.setRange(0.1, 100.0)
        self.ratio_input.setValue(5.0)
        self.ratio_input.setSingleStep(0.5)
        self.ratio_input.setSuffix(" %")
        settings_layout.addWidget(self.ratio_input, 2, 1)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("샘플링 시작")
        ok_btn.clicked.connect(self.dialog.accept)
        ok_btn.setStyleSheet("QPushButton { background-color: #0e639c; padding: 8px 20px; }")
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.dialog.reject)
        cancel_btn.setStyleSheet("QPushButton { background-color: #555; padding: 8px 20px; }")
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.dialog.setLayout(layout)
    
    def exec(self):
        return self.dialog.exec()
    
    def get_settings(self):
        """설정값 반환"""
        return {
            "layer": self.layer_combo.currentData(),
            "category": self.category_combo.currentIndex(),  # 0: both, 1: normal, 2: defect
            "ratio": self.ratio_input.value() / 100.0
        }


def main():
    app = QApplication(sys.argv)
    window = LayerSamplerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
