#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defect Viewer - 결함 이미지 확인 및 분류 도구

결함 이미지를 확인하고 2가지 유형으로 분류할 수 있는 GUI 도구
"""

import sys
import json
import shutil
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QTextEdit, QMessageBox, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QProgressBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush
import cv2
import numpy as np


class ZoomableGraphicsView(QGraphicsView):
    """확대/축소 가능한 이미지 뷰어"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self._zoom = 0

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        
        if -10 <= self._zoom <= 10:
            self.scale(factor, factor)
        else:
            self._zoom = max(-10, min(10, self._zoom))

    def reset_zoom(self):
        self.resetTransform()
        self._zoom = 0


class DefectViewerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.defect_dir = Path("D:/LLM_Dataset/output/Defect Layer")
        self.all_images = []
        self.current_index = 0
        self.classifications = {}  # {파일경로: 분류}
        self.classification_file = Path("D:/LLM_Dataset/defect_classifications.json")
        
        # 분류 카테고리 (수정 가능)
        self.categories = ["미분류", "유형1: 선형결함", "유형2: 면적결함"]
        
        self.init_ui()
        self.load_classifications()
        self.load_images()
    
    def init_ui(self):
        self.setWindowTitle("Defect Viewer - 결함 이미지 확인 및 분류")
        self.setGeometry(100, 100, 1400, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        
        # 왼쪽: 이미지 목록
        left_panel = QGroupBox("결함 이미지 목록")
        left_layout = QVBoxLayout()
        
        # 레이어 필터
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("레이어:"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("전체")
        for i in range(1, 19):
            self.layer_combo.addItem(f"Layer_{i:02d}")
        self.layer_combo.currentTextChanged.connect(self.filter_images)
        filter_layout.addWidget(self.layer_combo)
        
        filter_layout.addWidget(QLabel("분류:"))
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("전체")
        for cat in self.categories:
            self.class_filter_combo.addItem(cat)
        self.class_filter_combo.currentTextChanged.connect(self.filter_images)
        filter_layout.addWidget(self.class_filter_combo)
        
        left_layout.addLayout(filter_layout)
        
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        left_layout.addWidget(self.image_list)
        
        self.count_label = QLabel("0 / 0")
        left_layout.addWidget(self.count_label)
        
        left_panel.setLayout(left_layout)
        
        # 중앙: 이미지 뷰어
        center_panel = QGroupBox("이미지 뷰어 (휠: 확대/축소, 드래그: 이동)")
        center_layout = QVBoxLayout()
        
        self.scene = QGraphicsScene()
        self.graphics_view = ZoomableGraphicsView()
        self.graphics_view.setScene(self.scene)
        center_layout.addWidget(self.graphics_view)
        
        self.image_info_label = QLabel("이미지 정보")
        self.image_info_label.setWordWrap(True)
        center_layout.addWidget(self.image_info_label)
        
        center_panel.setLayout(center_layout)
        
        # 오른쪽: 분류 및 정보
        right_panel = QGroupBox("분류")
        right_layout = QVBoxLayout()
        
        # 분류 버튼들
        right_layout.addWidget(QLabel("결함 유형 선택:"))
        
        self.class_buttons = []
        for i, cat in enumerate(self.categories):
            btn = QPushButton(f"{i}: {cat}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, c=cat: self.classify_current(c))
            btn.setStyleSheet("QPushButton { padding: 15px; font-size: 14px; }")
            self.class_buttons.append(btn)
            right_layout.addWidget(btn)
        
        right_layout.addWidget(QLabel("\n키보드 단축키:"))
        right_layout.addWidget(QLabel("0: 미분류"))
        right_layout.addWidget(QLabel("1: 유형1 (선형결함)"))
        right_layout.addWidget(QLabel("2: 유형2 (면적결함)"))
        right_layout.addWidget(QLabel("← →: 이전/다음 이미지"))
        right_layout.addWidget(QLabel("R: 확대/축소 초기화"))
        
        # 현재 분류 상태
        self.current_class_label = QLabel("\n현재 분류: 미분류")
        self.current_class_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(self.current_class_label)
        
        # 통계
        stats_group = QGroupBox("분류 통계")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("미분류: 0\n유형1: 0\n유형2: 0")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # 저장/내보내기 버튼
        save_btn = QPushButton("💾 분류 결과 저장")
        save_btn.clicked.connect(self.save_classifications)
        right_layout.addWidget(save_btn)
        
        export_btn = QPushButton("📁 분류별 폴더로 복사")
        export_btn.clicked.connect(self.export_to_folders)
        right_layout.addWidget(export_btn)
        
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        
        # 스플리터로 배치
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        
        # 스타일
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #d4d4d4; }
            QGroupBox { border: 2px solid #3c3c3c; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QListWidget { background-color: #252526; border: 1px solid #3c3c3c; }
            QListWidget::item:selected { background-color: #094771; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555; padding: 8px; }
            QPushButton:hover { background-color: #4c4c4c; }
            QPushButton:checked { background-color: #0e639c; }
            QComboBox { background-color: #3c3c3c; border: 1px solid #555; padding: 5px; }
        """)
    
    def keyPressEvent(self, event):
        """키보드 단축키"""
        key = event.key()
        
        if key == Qt.Key_0:
            self.classify_current(self.categories[0])
        elif key == Qt.Key_1:
            self.classify_current(self.categories[1])
        elif key == Qt.Key_2:
            self.classify_current(self.categories[2])
        elif key == Qt.Key_Left:
            self.prev_image()
        elif key == Qt.Key_Right:
            self.next_image()
        elif key == Qt.Key_R:
            self.graphics_view.reset_zoom()
    
    def load_images(self):
        """결함 이미지 로드"""
        self.all_images = []
        
        for layer_folder in sorted(self.defect_dir.iterdir()):
            if layer_folder.is_dir():
                for img_path in sorted(layer_folder.glob("*.jpg")):
                    self.all_images.append(img_path)
                for img_path in sorted(layer_folder.glob("*.png")):
                    self.all_images.append(img_path)
        
        self.filter_images()
        self.update_stats()
    
    def filter_images(self):
        """이미지 필터링"""
        layer_filter = self.layer_combo.currentText()
        class_filter = self.class_filter_combo.currentText()
        
        self.image_list.clear()
        filtered_count = 0
        
        for img_path in self.all_images:
            # 레이어 필터
            if layer_filter != "전체":
                if layer_filter not in str(img_path.parent.name):
                    continue
            
            # 분류 필터
            if class_filter != "전체":
                img_class = self.classifications.get(str(img_path), "미분류")
                if img_class != class_filter:
                    continue
            
            item = QListWidgetItem(f"{img_path.parent.name}/{img_path.name}")
            item.setData(Qt.UserRole, str(img_path))
            
            # 분류에 따른 색상
            img_class = self.classifications.get(str(img_path), "미분류")
            if img_class == self.categories[1]:
                item.setBackground(QColor(50, 100, 50))
            elif img_class == self.categories[2]:
                item.setBackground(QColor(100, 50, 50))
            
            self.image_list.addItem(item)
            filtered_count += 1
        
        self.count_label.setText(f"{filtered_count} / {len(self.all_images)}")
        
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
    
    def on_image_selected(self, row):
        """이미지 선택 시"""
        if row < 0:
            return
        
        item = self.image_list.item(row)
        img_path = Path(item.data(Qt.UserRole))
        
        self.display_image(img_path)
        self.update_class_buttons(img_path)
    
    def display_image(self, img_path):
        """이미지 표시"""
        try:
            # 한글 경로 처리
            with open(img_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            if img is None:
                return
            
            # BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            
            # QPixmap으로 변환
            from PyQt5.QtGui import QImage
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scene에 표시
            self.scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)
            self.scene.setSceneRect(pixmap_item.boundingRect())
            
            # 뷰에 맞추기
            self.graphics_view.reset_zoom()
            self.graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)
            
            # 이미지 정보
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w) * 100
            
            self.image_info_label.setText(
                f"파일: {img_path.name}\n"
                f"크기: {w}x{h} | 평균밝기: {mean_val:.1f} | 표준편차: {std_val:.1f} | 엣지밀도: {edge_density:.2f}%"
            )
            
        except Exception as e:
            self.image_info_label.setText(f"오류: {e}")
    
    def update_class_buttons(self, img_path):
        """분류 버튼 상태 업데이트"""
        current_class = self.classifications.get(str(img_path), "미분류")
        
        for i, btn in enumerate(self.class_buttons):
            btn.setChecked(self.categories[i] == current_class)
        
        self.current_class_label.setText(f"\n현재 분류: {current_class}")
    
    def classify_current(self, category):
        """현재 이미지 분류"""
        row = self.image_list.currentRow()
        if row < 0:
            return
        
        item = self.image_list.item(row)
        img_path = item.data(Qt.UserRole)
        
        self.classifications[img_path] = category
        
        # 색상 업데이트
        if category == self.categories[1]:
            item.setBackground(QColor(50, 100, 50))
        elif category == self.categories[2]:
            item.setBackground(QColor(100, 50, 50))
        else:
            item.setBackground(QColor(37, 37, 38))
        
        self.update_class_buttons(Path(img_path))
        self.update_stats()
        
        # 자동으로 다음 이미지로
        self.next_image()
    
    def prev_image(self):
        """이전 이미지"""
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)
    
    def next_image(self):
        """다음 이미지"""
        row = self.image_list.currentRow()
        if row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(row + 1)
    
    def update_stats(self):
        """통계 업데이트"""
        stats = {cat: 0 for cat in self.categories}
        
        for img_class in self.classifications.values():
            if img_class in stats:
                stats[img_class] += 1
        
        # 미분류 수 계산
        classified = sum(stats.values()) - stats.get("미분류", 0)
        stats["미분류"] = len(self.all_images) - classified
        
        text = "\n".join([f"{cat}: {count}" for cat, count in stats.items()])
        self.stats_label.setText(text)
    
    def load_classifications(self):
        """저장된 분류 로드"""
        if self.classification_file.exists():
            try:
                with open(self.classification_file, 'r', encoding='utf-8') as f:
                    self.classifications = json.load(f)
                print(f"분류 로드: {len(self.classifications)}개")
            except:
                self.classifications = {}
    
    def save_classifications(self):
        """분류 저장"""
        try:
            with open(self.classification_file, 'w', encoding='utf-8') as f:
                json.dump(self.classifications, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "저장 완료", f"분류 결과가 저장되었습니다.\n{self.classification_file}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"저장 실패: {e}")
    
    def export_to_folders(self):
        """분류별 폴더로 복사"""
        output_dir = QFileDialog.getExistingDirectory(self, "출력 폴더 선택")
        if not output_dir:
            return
        
        output_path = Path(output_dir)
        
        try:
            # 분류별 폴더 생성
            for cat in self.categories[1:]:  # 미분류 제외
                cat_dir = output_path / cat.replace(":", "_").replace(" ", "_")
                cat_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일 복사
            copied = 0
            for img_path_str, category in self.classifications.items():
                if category == "미분류":
                    continue
                
                img_path = Path(img_path_str)
                if img_path.exists():
                    cat_dir = output_path / category.replace(":", "_").replace(" ", "_")
                    dst = cat_dir / img_path.name
                    shutil.copy2(img_path, dst)
                    copied += 1
            
            QMessageBox.information(self, "완료", f"{copied}개 파일이 복사되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"내보내기 실패: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DefectViewerGUI()
    window.show()
    sys.exit(app.exec_())
