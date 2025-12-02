#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classifier GUI - 폴더 간 이미지 분류 도구

기능:
1. 좌/우 폴더 선택 가능
2. 폴더 간 이미지 이동
3. 진행 상황 표시 (번호/전체)
4. 이동 후 위치 유지
5. Undo 기능
"""

import sys
import os
import shutil
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QSplitter,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGroupBox,
    QMessageBox, QShortcut, QFrame, QComboBox
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QKeySequence, QFont, QColor, QPainter


class ZoomableGraphicsView(QGraphicsView):
    """줌 및 패닝 지원 이미지 뷰어"""
    
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
        
        self._zoom = 1.0
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._rotation = 0
    
    def set_image(self, image_path):
        """이미지 로드 및 표시"""
        self._scene.clear()
        self._rotation = 0
        
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                self._pixmap_item = QGraphicsPixmapItem(pixmap)
                self._scene.addItem(self._pixmap_item)
                self._scene.setSceneRect(QRectF(pixmap.rect()))
                self.fit_in_view()
                return True
        return False
    
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
    
    def rotate_image(self, angle):
        """이미지 회전"""
        self._rotation += angle
        self.rotate(angle)
    
    def reset_view(self):
        """뷰 리셋"""
        self.resetTransform()
        self._rotation = 0
        self.fit_in_view()
    
    def mouseDoubleClickEvent(self, event):
        """더블클릭으로 뷰 리셋"""
        self.reset_view()
        super().mouseDoubleClickEvent(event)


class ImageClassifierGUI(QMainWindow):
    """이미지 분류 메인 GUI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier - 폴더 간 이미지 분류")
        self.setGeometry(100, 100, 1600, 900)
        
        # 기본 폴더 경로
        self.base_path = Path(r"D:\LLM_Dataset\output")
        
        # 사용 가능한 폴더 목록
        self.available_folders = self.scan_folders()
        
        # 기본 선택 폴더
        self.left_folder = None
        self.right_folder = None
        
        # 이미지 리스트
        self.left_images = []
        self.right_images = []
        
        # 현재 선택 상태
        self.current_source = None  # 'left' or 'right'
        self.current_index = -1
        
        # Undo 스택
        self.undo_stack = []
        self.max_undo = 50
        
        self.init_ui()
        self.setup_shortcuts()
        
        # 기본 폴더 설정
        self.set_default_folders()
    
    def scan_folders(self):
        """output 폴더 내 하위 폴더 스캔"""
        folders = []
        if self.base_path.exists():
            for item in sorted(self.base_path.iterdir()):
                if item.is_dir():
                    folders.append(item.name)
        return folders
    
    def set_default_folders(self):
        """기본 폴더 설정"""
        # 기본값 설정
        if "Normal Image" in self.available_folders:
            idx = self.available_folders.index("Normal Image")
            self.left_combo.setCurrentIndex(idx)
        elif len(self.available_folders) > 0:
            self.left_combo.setCurrentIndex(0)
        
        if "Twist Image" in self.available_folders:
            idx = self.available_folders.index("Twist Image")
            self.right_combo.setCurrentIndex(idx)
        elif len(self.available_folders) > 1:
            self.right_combo.setCurrentIndex(1)
    
    def init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 상단: 폴더 선택 영역
        folder_layout = QHBoxLayout()
        
        # 왼쪽 폴더 선택
        folder_layout.addWidget(QLabel("왼쪽 폴더:"))
        self.left_combo = QComboBox()
        self.left_combo.addItems(self.available_folders)
        self.left_combo.currentTextChanged.connect(self.on_left_folder_changed)
        self.left_combo.setMinimumWidth(200)
        folder_layout.addWidget(self.left_combo)
        
        folder_layout.addSpacing(50)
        
        # 오른쪽 폴더 선택
        folder_layout.addWidget(QLabel("오른쪽 폴더:"))
        self.right_combo = QComboBox()
        self.right_combo.addItems(self.available_folders)
        self.right_combo.currentTextChanged.connect(self.on_right_folder_changed)
        self.right_combo.setMinimumWidth(200)
        folder_layout.addWidget(self.right_combo)
        
        folder_layout.addStretch()
        
        # 폴더 새로고침 버튼
        refresh_btn = QPushButton("🔄 폴더 새로고침")
        refresh_btn.clicked.connect(self.refresh_folders)
        folder_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(folder_layout)
        
        # 3열 레이아웃: Left | Viewer | Right
        splitter = QSplitter(Qt.Horizontal)
        
        # 왼쪽 패널
        self.left_panel = self.create_list_panel("left")
        splitter.addWidget(self.left_panel)
        
        # 중앙: 이미지 뷰어
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # 오른쪽 패널
        self.right_panel = self.create_list_panel("right")
        splitter.addWidget(self.right_panel)
        
        # 비율 설정 (1:2:1)
        splitter.setSizes([300, 800, 300])
        
        main_layout.addWidget(splitter)
        
        # 다크 테마 스타일 적용
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
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px 10px;
                border-radius: 3px;
                color: #d4d4d4;
            }
            QComboBox:hover {
                border: 1px solid #0e639c;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                border: 1px solid #555;
                selection-background-color: #0e639c;
            }
        """)
    
    def on_left_folder_changed(self, folder_name):
        """왼쪽 폴더 변경"""
        if folder_name:
            self.left_folder = self.base_path / folder_name
            self.left_panel.setTitle(folder_name)
            self.load_images()
    
    def on_right_folder_changed(self, folder_name):
        """오른쪽 폴더 변경"""
        if folder_name:
            self.right_folder = self.base_path / folder_name
            self.right_panel.setTitle(folder_name)
            self.load_images()
    
    def refresh_folders(self):
        """폴더 목록 새로고침"""
        current_left = self.left_combo.currentText()
        current_right = self.right_combo.currentText()
        
        self.available_folders = self.scan_folders()
        
        self.left_combo.blockSignals(True)
        self.right_combo.blockSignals(True)
        
        self.left_combo.clear()
        self.right_combo.clear()
        self.left_combo.addItems(self.available_folders)
        self.right_combo.addItems(self.available_folders)
        
        # 이전 선택 복원
        if current_left in self.available_folders:
            self.left_combo.setCurrentText(current_left)
        if current_right in self.available_folders:
            self.right_combo.setCurrentText(current_right)
        
        self.left_combo.blockSignals(False)
        self.right_combo.blockSignals(False)
        
        self.load_images()
        self.update_status("폴더 목록 새로고침 완료")
    
    def create_list_panel(self, list_type):
        """리스트 패널 생성"""
        panel = QGroupBox("폴더 선택 필요")
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        
        # 진행 상황 레이블
        progress_label = QLabel("0 / 0")
        progress_label.setAlignment(Qt.AlignCenter)
        progress_label.setStyleSheet("QLabel { font-weight: bold; color: #4a9eff; padding: 5px; }")
        layout.addWidget(progress_label)
        
        # 리스트 위젯
        list_widget = QListWidget()
        list_widget.setFont(QFont("Consolas", 9))
        layout.addWidget(list_widget)
        
        # 버튼 영역
        btn_layout = QHBoxLayout()
        
        if list_type == "left":
            # Left → Right 이동 버튼
            move_btn = QPushButton("→ 오른쪽으로 이동 (→)")
            move_btn.clicked.connect(self.move_to_right)
            move_btn.setStyleSheet("""
                QPushButton {
                    background-color: #b8860b;
                    padding: 8px 15px;
                }
                QPushButton:hover {
                    background-color: #daa520;
                }
            """)
            btn_layout.addWidget(move_btn)
            
            # 위젯 참조 저장
            self.left_list = list_widget
            self.left_progress_label = progress_label
            self.left_move_btn = move_btn
            list_widget.currentRowChanged.connect(self.on_left_selection_changed)
        else:
            # Right → Left 이동 버튼
            move_btn = QPushButton("← 왼쪽으로 이동 (←)")
            move_btn.clicked.connect(self.move_to_left)
            move_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0a7a0a;
                    padding: 8px 15px;
                }
                QPushButton:hover {
                    background-color: #0c9a0c;
                }
            """)
            btn_layout.addWidget(move_btn)
            
            # 위젯 참조 저장
            self.right_list = list_widget
            self.right_progress_label = progress_label
            self.right_move_btn = move_btn
            list_widget.currentRowChanged.connect(self.on_right_selection_changed)
        
        layout.addLayout(btn_layout)
        
        return panel
    
    def create_center_panel(self):
        """중앙 이미지 뷰어 패널"""
        panel = QGroupBox("이미지 미리보기")
        layout = QVBoxLayout(panel)
        
        # 현재 이미지 정보
        self.image_info_label = QLabel("이미지를 선택하세요 | 마우스 휠: 확대/축소 | 드래그: 이동 | 더블클릭: 초기화")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setStyleSheet("QLabel { color: #888; padding: 5px; }")
        self.image_info_label.setWordWrap(True)
        layout.addWidget(self.image_info_label)
        
        # 이미지 뷰어
        self.image_viewer = ZoomableGraphicsView()
        layout.addWidget(self.image_viewer)
        
        # 컨트롤 버튼들
        control_layout = QHBoxLayout()
        
        # 회전 버튼
        rotate_left_btn = QPushButton("↶ 회전")
        rotate_left_btn.clicked.connect(lambda: self.image_viewer.rotate_image(-90))
        control_layout.addWidget(rotate_left_btn)
        
        rotate_right_btn = QPushButton("회전 ↷")
        rotate_right_btn.clicked.connect(lambda: self.image_viewer.rotate_image(90))
        control_layout.addWidget(rotate_right_btn)
        
        # 맞춤 버튼
        fit_btn = QPushButton("화면 맞춤")
        fit_btn.clicked.connect(self.image_viewer.reset_view)
        control_layout.addWidget(fit_btn)
        
        control_layout.addStretch()
        
        # Undo 버튼
        self.undo_btn = QPushButton("↩ 실행취소 (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_action)
        self.undo_btn.setEnabled(False)
        control_layout.addWidget(self.undo_btn)
        
        # 삭제 버튼
        delete_btn = QPushButton("🗑 삭제 (Delete)")
        delete_btn.clicked.connect(self.delete_selected)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a00000;
            }
        """)
        control_layout.addWidget(delete_btn)
        
        layout.addLayout(control_layout)
        
        # 도움말
        help_text = QLabel("단축키: ←→ 이동 | ↑↓ 탐색 | 1/2 폴더 포커스 | Ctrl+Z 실행취소 | Delete 삭제 | Space 화면맞춤")
        help_text.setAlignment(Qt.AlignCenter)
        help_text.setStyleSheet("QLabel { color: #666; font-size: 9pt; padding: 5px; }")
        layout.addWidget(help_text)
        
        # 상태 레이블
        self.status_label = QLabel("준비 완료")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { color: #4a9eff; font-weight: bold; padding: 5px; background-color: #2a2a2a; border-radius: 3px; }")
        layout.addWidget(self.status_label)
        
        return panel
    
    def setup_shortcuts(self):
        """키보드 단축키 설정"""
        # 이동 단축키
        QShortcut(QKeySequence(Qt.Key_Left), self, self.move_to_left)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.move_to_right)
        
        # 탐색 단축키
        QShortcut(QKeySequence(Qt.Key_Up), self, self.select_previous)
        QShortcut(QKeySequence(Qt.Key_Down), self, self.select_next)
        
        # 폴더 포커스 단축키
        QShortcut(QKeySequence(Qt.Key_1), self, lambda: self.focus_list('left'))
        QShortcut(QKeySequence(Qt.Key_2), self, lambda: self.focus_list('right'))
        
        # Undo 단축키
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_action)
        
        # 삭제 단축키
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected)
        
        # 화면 맞춤
        QShortcut(QKeySequence(Qt.Key_Space), self, self.image_viewer.reset_view)
    
    def load_images(self):
        """이미지 파일 로드"""
        self.left_images = []
        self.right_images = []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # 왼쪽 이미지 로드
        if self.left_folder and self.left_folder.exists():
            for f in sorted(self.left_folder.iterdir()):
                if f.suffix.lower() in image_extensions:
                    self.left_images.append(f)
        
        # 오른쪽 이미지 로드
        if self.right_folder and self.right_folder.exists():
            for f in sorted(self.right_folder.iterdir()):
                if f.suffix.lower() in image_extensions:
                    self.right_images.append(f)
        
        self.update_lists()
        self.update_status(f"로드 완료: 왼쪽 {len(self.left_images)}개 | 오른쪽 {len(self.right_images)}개")
    
    def update_lists(self, restore_selection=None):
        """리스트 위젯 업데이트"""
        # 현재 선택 저장
        if restore_selection is None and self.current_source:
            restore_selection = (self.current_source, self.current_index)
        
        # 왼쪽 리스트 업데이트
        self.left_list.blockSignals(True)
        self.left_list.clear()
        total_left = len(self.left_images)
        for i, img_path in enumerate(self.left_images):
            item_text = f"{i+1:04d}/{total_left}: {img_path.name}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, str(img_path))
            self.left_list.addItem(item)
        self.left_list.blockSignals(False)
        
        # 왼쪽 진행 표시
        self.left_progress_label.setText(f"총 {total_left}개")
        
        # 오른쪽 리스트 업데이트
        self.right_list.blockSignals(True)
        self.right_list.clear()
        total_right = len(self.right_images)
        for i, img_path in enumerate(self.right_images):
            item_text = f"{i+1:04d}/{total_right}: {img_path.name}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, str(img_path))
            self.right_list.addItem(item)
        self.right_list.blockSignals(False)
        
        # 오른쪽 진행 표시
        self.right_progress_label.setText(f"총 {total_right}개")
        
        # 선택 복원
        if restore_selection:
            source, index = restore_selection
            if source == 'left':
                if len(self.left_images) > 0:
                    index = min(index, len(self.left_images) - 1)
                    index = max(0, index)
                    self.left_list.setCurrentRow(index)
                elif len(self.right_images) > 0:
                    self.right_list.setCurrentRow(0)
            else:
                if len(self.right_images) > 0:
                    index = min(index, len(self.right_images) - 1)
                    index = max(0, index)
                    self.right_list.setCurrentRow(index)
                elif len(self.left_images) > 0:
                    self.left_list.setCurrentRow(0)
    
    def on_left_selection_changed(self, row):
        """왼쪽 리스트 선택 변경"""
        if row >= 0 and row < len(self.left_images):
            self.right_list.blockSignals(True)
            self.right_list.clearSelection()
            self.right_list.setCurrentRow(-1)
            self.right_list.blockSignals(False)
            
            self.current_source = 'left'
            self.current_index = row
            self.display_image(self.left_images[row])
            
            # 진행 표시 업데이트
            self.left_progress_label.setText(f"{row+1} / {len(self.left_images)}")
            self.left_progress_label.setStyleSheet("QLabel { font-weight: bold; color: #4a9eff; padding: 5px; }")
            self.right_progress_label.setText(f"총 {len(self.right_images)}개")
            self.right_progress_label.setStyleSheet("QLabel { font-weight: bold; color: #888; padding: 5px; }")
    
    def on_right_selection_changed(self, row):
        """오른쪽 리스트 선택 변경"""
        if row >= 0 and row < len(self.right_images):
            self.left_list.blockSignals(True)
            self.left_list.clearSelection()
            self.left_list.setCurrentRow(-1)
            self.left_list.blockSignals(False)
            
            self.current_source = 'right'
            self.current_index = row
            self.display_image(self.right_images[row])
            
            # 진행 표시 업데이트
            self.right_progress_label.setText(f"{row+1} / {len(self.right_images)}")
            self.right_progress_label.setStyleSheet("QLabel { font-weight: bold; color: #ff6b6b; padding: 5px; }")
            self.left_progress_label.setText(f"총 {len(self.left_images)}개")
            self.left_progress_label.setStyleSheet("QLabel { font-weight: bold; color: #888; padding: 5px; }")
    
    def display_image(self, image_path):
        """이미지 표시"""
        if self.image_viewer.set_image(str(image_path)):
            file_size = image_path.stat().st_size / 1024
            folder_name = self.left_combo.currentText() if self.current_source == 'left' else self.right_combo.currentText()
            total = len(self.left_images) if self.current_source == 'left' else len(self.right_images)
            
            info_text = f"[{folder_name}] {self.current_index + 1}/{total}: {image_path.name} ({file_size:.1f} KB)"
            self.image_info_label.setText(info_text)
            
            if self.current_source == 'left':
                self.image_info_label.setStyleSheet("QLabel { color: #4a9eff; padding: 5px; font-weight: bold; }")
            else:
                self.image_info_label.setStyleSheet("QLabel { color: #ff6b6b; padding: 5px; font-weight: bold; }")
    
    def move_to_right(self):
        """왼쪽 → 오른쪽 이동"""
        if self.current_source != 'left' or self.current_index < 0:
            self.update_status("왼쪽 리스트에서 이미지를 선택하세요")
            return
        
        if not self.right_folder:
            self.update_status("오른쪽 폴더를 선택하세요")
            return
        
        if self.current_index >= len(self.left_images):
            return
        
        src_path = self.left_images[self.current_index]
        dst_path = self.right_folder / src_path.name
        
        # 파일 중복 확인
        if dst_path.exists():
            base = dst_path.stem
            ext = dst_path.suffix
            counter = 1
            while dst_path.exists():
                dst_path = self.right_folder / f"{base}_{counter}{ext}"
                counter += 1
        
        try:
            shutil.move(str(src_path), str(dst_path))
            
            # Undo 정보 저장
            self.add_undo_action('move', dst_path, self.left_folder, self.current_index, src_path.name)
            
            restore_index = self.current_index
            
            self.left_images.pop(self.current_index)
            self.right_images.append(dst_path)
            self.right_images.sort(key=lambda x: x.name)
            
            self.update_lists(restore_selection=('left', restore_index))
            self.update_status(f"→ 오른쪽으로 이동: {src_path.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이동 실패: {e}")
    
    def move_to_left(self):
        """오른쪽 → 왼쪽 이동"""
        if self.current_source != 'right' or self.current_index < 0:
            self.update_status("오른쪽 리스트에서 이미지를 선택하세요")
            return
        
        if not self.left_folder:
            self.update_status("왼쪽 폴더를 선택하세요")
            return
        
        if self.current_index >= len(self.right_images):
            return
        
        src_path = self.right_images[self.current_index]
        dst_path = self.left_folder / src_path.name
        
        if dst_path.exists():
            base = dst_path.stem
            ext = dst_path.suffix
            counter = 1
            while dst_path.exists():
                dst_path = self.left_folder / f"{base}_{counter}{ext}"
                counter += 1
        
        try:
            shutil.move(str(src_path), str(dst_path))
            
            self.add_undo_action('move', dst_path, self.right_folder, self.current_index, src_path.name)
            
            restore_index = self.current_index
            
            self.right_images.pop(self.current_index)
            self.left_images.append(dst_path)
            self.left_images.sort(key=lambda x: x.name)
            
            self.update_lists(restore_selection=('right', restore_index))
            self.update_status(f"← 왼쪽으로 이동: {src_path.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이동 실패: {e}")
    
    def add_undo_action(self, action_type, current_path, original_folder, original_index, original_name):
        """Undo 스택에 액션 추가"""
        self.undo_stack.append({
            'type': action_type,
            'current_path': current_path,
            'original_folder': original_folder,
            'original_index': original_index,
            'original_name': original_name
        })
        
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        
        self.undo_btn.setEnabled(True)
        self.undo_btn.setText(f"↩ 실행취소 ({len(self.undo_stack)})")
    
    def undo_action(self):
        """마지막 액션 취소"""
        if not self.undo_stack:
            self.update_status("취소할 작업이 없습니다")
            return
        
        action = self.undo_stack.pop()
        
        if action['type'] == 'move':
            src_path = action['current_path']
            dst_folder = action['original_folder']
            original_name = action['original_name']
            
            dst_path = dst_folder / original_name
            
            try:
                if src_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    self.load_images()
                    
                    if dst_folder == self.left_folder:
                        self.focus_list('left')
                        idx = next((i for i, p in enumerate(self.left_images) if p.name == original_name), 0)
                        self.left_list.setCurrentRow(idx)
                    else:
                        self.focus_list('right')
                        idx = next((i for i, p in enumerate(self.right_images) if p.name == original_name), 0)
                        self.right_list.setCurrentRow(idx)
                    
                    self.update_status(f"↩ 실행취소: {original_name}")
                    
            except Exception as e:
                QMessageBox.critical(self, "오류", f"실행취소 실패: {e}")
        
        if self.undo_stack:
            self.undo_btn.setText(f"↩ 실행취소 ({len(self.undo_stack)})")
        else:
            self.undo_btn.setEnabled(False)
            self.undo_btn.setText("↩ 실행취소 (Ctrl+Z)")
    
    def delete_selected(self):
        """선택된 이미지 삭제"""
        if not self.current_source or self.current_index < 0:
            self.update_status("삭제할 이미지를 선택하세요")
            return
        
        if self.current_source == 'left':
            if self.current_index >= len(self.left_images):
                return
            image_path = self.left_images[self.current_index]
        else:
            if self.current_index >= len(self.right_images):
                return
            image_path = self.right_images[self.current_index]
        
        reply = QMessageBox.question(
            self, "삭제 확인",
            f"정말 삭제하시겠습니까?\n\n{image_path.name}\n\n⚠️ 이 작업은 취소할 수 없습니다!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                restore_index = self.current_index
                restore_source = self.current_source
                
                os.remove(str(image_path))
                
                if self.current_source == 'left':
                    self.left_images.pop(self.current_index)
                else:
                    self.right_images.pop(self.current_index)
                
                self.update_lists(restore_selection=(restore_source, restore_index))
                self.update_status(f"🗑 삭제됨: {image_path.name}")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"삭제 실패: {e}")
    
    def select_previous(self):
        """이전 항목 선택"""
        if self.current_source == 'left':
            if self.current_index > 0:
                self.left_list.setCurrentRow(self.current_index - 1)
        elif self.current_source == 'right':
            if self.current_index > 0:
                self.right_list.setCurrentRow(self.current_index - 1)
    
    def select_next(self):
        """다음 항목 선택"""
        if self.current_source == 'left':
            if self.current_index < len(self.left_images) - 1:
                self.left_list.setCurrentRow(self.current_index + 1)
        elif self.current_source == 'right':
            if self.current_index < len(self.right_images) - 1:
                self.right_list.setCurrentRow(self.current_index + 1)
    
    def focus_list(self, list_type):
        """리스트에 포커스"""
        if list_type == 'left':
            self.left_list.setFocus()
            if self.left_list.currentRow() < 0 and len(self.left_images) > 0:
                self.left_list.setCurrentRow(0)
        else:
            self.right_list.setFocus()
            if self.right_list.currentRow() < 0 and len(self.right_images) > 0:
                self.right_list.setCurrentRow(0)
    
    def update_status(self, message):
        """상태 업데이트"""
        total = len(self.left_images) + len(self.right_images)
        left_name = self.left_combo.currentText() if self.left_folder else "미선택"
        right_name = self.right_combo.currentText() if self.right_folder else "미선택"
        self.status_label.setText(f"{message} | {left_name}: {len(self.left_images)} | {right_name}: {len(self.right_images)} | Total: {total}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ImageClassifierGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
