#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Mover - 분류 결과 기반 이미지 이동 도구

기능:
1. JSON 분류 파일 로드
2. 원본 폴더에서 이미지 검색
3. 분류(Normal, Twist, Hook)별 서브폴더 생성
4. 이미지를 해당 분류 폴더로 이동/복사
"""

import sys
import os
import json
import shutil
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QListWidget, QListWidgetItem, QGroupBox, QRadioButton, QButtonGroup,
    QCheckBox, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor


class MoverThread(QThread):
    """파일 이동/복사 스레드"""
    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(int, int, int)  # success, failed, skipped
    error = pyqtSignal(str)
    log = pyqtSignal(str)  # 로그 메시지
    
    def __init__(self, classification_data, search_root, move_mode=True):
        super().__init__()
        self.classification_data = classification_data
        self.search_root = Path(search_root)
        self.move_mode = move_mode  # True: 이동, False: 복사
        self._stop = False
    
    def stop(self):
        self._stop = True
    
    def run(self):
        try:
            images = self.classification_data.get("images", [])
            source_folder_name = Path(self.classification_data["metadata"]["source_folder"]).name
            
            # 소스 폴더 찾기
            source_folder = self.find_folder(source_folder_name)
            
            if not source_folder:
                self.error.emit(f"폴더를 찾을 수 없습니다: {source_folder_name}")
                return
            
            self.log.emit(f"📁 소스 폴더 발견: {source_folder}")
            
            # 분류별 서브폴더 생성
            class_names = self.classification_data["metadata"].get("class_names", ["Normal", "Twist", "Hook"])
            class_folders = {}
            
            for class_name in class_names:
                class_folder = source_folder / class_name
                class_folder.mkdir(exist_ok=True)
                class_folders[class_name] = class_folder
                self.log.emit(f"📂 폴더 생성: {class_folder}")
            
            total = len(images)
            success = 0
            failed = 0
            skipped = 0
            
            for i, image_info in enumerate(images):
                if self._stop:
                    self.log.emit("⚠️ 사용자에 의해 중지됨")
                    break
                
                filename = image_info["filename"]
                predicted_class = image_info["predicted_class"]
                
                self.progress.emit(i + 1, total, filename)
                
                # 원본 파일 찾기
                source_file = source_folder / filename
                
                if not source_file.exists():
                    self.log.emit(f"⚠️ 파일 없음: {filename}")
                    skipped += 1
                    continue
                
                # 대상 폴더
                if predicted_class not in class_folders:
                    self.log.emit(f"⚠️ 알 수 없는 클래스: {predicted_class}")
                    skipped += 1
                    continue
                
                dest_folder = class_folders[predicted_class]
                dest_file = dest_folder / filename
                
                # 이미 대상 폴더에 있는 경우
                if dest_file.exists():
                    skipped += 1
                    continue
                
                try:
                    if self.move_mode:
                        shutil.move(str(source_file), str(dest_file))
                    else:
                        shutil.copy2(str(source_file), str(dest_file))
                    success += 1
                except Exception as e:
                    self.log.emit(f"❌ 실패 ({filename}): {e}")
                    failed += 1
                
                # UI 반응성
                self.msleep(1)
            
            self.finished.emit(success, failed, skipped)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def find_folder(self, folder_name):
        """검색 루트에서 폴더 이름으로 찾기"""
        # 정확히 일치하는 폴더 찾기
        for root, dirs, files in os.walk(self.search_root):
            for d in dirs:
                if d == folder_name:
                    return Path(root) / d
        return None


class ClassificationMoverGUI(QMainWindow):
    """분류 이동 메인 GUI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classification Mover - 분류 결과 기반 이미지 이동")
        self.setGeometry(100, 100, 1200, 800)
        
        # 데이터
        self.classification_data = None
        self.json_path = None
        self.search_root = Path(r"K:\LLM Image_Storage")  # 기본 검색 루트
        
        # 스레드
        self.mover_thread = None
        
        # 자동화 관련
        self.automation_json_files = []  # 자동화 대상 JSON 파일 목록
        self.automation_index = 0  # 현재 처리 중인 인덱스
        self.is_automation_running = False
        self.automation_results = []  # 자동화 결과 저장
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # JSON 파일 선택
        json_group = QGroupBox("1. 분류 JSON 파일 선택")
        json_layout = QHBoxLayout(json_group)
        
        self.json_path_label = QLabel("JSON 파일을 선택하세요")
        self.json_path_label.setStyleSheet("QLabel { color: #888; padding: 5px; }")
        json_layout.addWidget(self.json_path_label, 1)
        
        self.load_json_btn = QPushButton("📂 JSON 파일 선택")
        self.load_json_btn.clicked.connect(self.load_json_file)
        json_layout.addWidget(self.load_json_btn)
        
        self.load_folder_btn = QPushButton("📁 JSON 폴더 열기")
        self.load_folder_btn.clicked.connect(self.open_json_folder)
        json_layout.addWidget(self.load_folder_btn)
        
        # 자동화 버튼 추가
        self.auto_btn = QPushButton("⚡ 자동화")
        self.auto_btn.setToolTip("여러 JSON 파일을 선택하여 순차적으로 처리")
        self.auto_btn.clicked.connect(self.start_automation)
        self.auto_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6f00;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #ff8f00;
            }
        """)
        json_layout.addWidget(self.auto_btn)
        
        main_layout.addWidget(json_group)
        
        # 검색 루트 설정
        search_group = QGroupBox("2. 이미지 검색 루트 폴더")
        search_layout = QHBoxLayout(search_group)
        
        self.search_root_edit = QLineEdit(str(self.search_root))
        self.search_root_edit.setStyleSheet("QLineEdit { padding: 5px; }")
        search_layout.addWidget(self.search_root_edit, 1)
        
        self.browse_search_btn = QPushButton("📂 변경")
        self.browse_search_btn.clicked.connect(self.browse_search_root)
        search_layout.addWidget(self.browse_search_btn)
        
        main_layout.addWidget(search_group)
        
        # 스플리터 (정보 + 로그)
        splitter = QSplitter(Qt.Horizontal)
        
        # 왼쪽: 분류 정보
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        info_group = QGroupBox("분류 정보")
        info_inner_layout = QVBoxLayout(info_group)
        
        # 메타데이터
        self.meta_label = QLabel("JSON 파일을 로드하세요")
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet("QLabel { background-color: #2a2a2a; padding: 10px; border-radius: 5px; }")
        info_inner_layout.addWidget(self.meta_label)
        
        # 통계 테이블
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(["클래스", "개수", "비율"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setMaximumHeight(150)
        info_inner_layout.addWidget(self.stats_table)
        
        info_layout.addWidget(info_group)
        splitter.addWidget(info_widget)
        
        # 오른쪽: 로그
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        log_group = QGroupBox("작업 로그")
        log_inner_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }")
        log_inner_layout.addWidget(self.log_text)
        
        log_layout.addWidget(log_group)
        splitter.addWidget(log_widget)
        
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)
        
        # 옵션
        option_group = QGroupBox("3. 작업 옵션")
        option_layout = QHBoxLayout(option_group)
        
        self.move_radio = QRadioButton("이동 (원본 삭제)")
        self.copy_radio = QRadioButton("복사 (원본 유지)")
        self.move_radio.setChecked(True)
        
        option_layout.addWidget(self.move_radio)
        option_layout.addWidget(self.copy_radio)
        option_layout.addStretch()
        
        main_layout.addWidget(option_group)
        
        # 진행바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 실행 버튼
        btn_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶ 분류별 이동 시작")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.start_moving)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                padding: 15px 30px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:disabled { background-color: #3c3c3c; }
        """)
        btn_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("⏹ 중지")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_moving)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                padding: 15px 30px;
                font-size: 14pt;
            }
            QPushButton:hover { background-color: #f44336; }
            QPushButton:disabled { background-color: #3c3c3c; }
        """)
        btn_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(btn_layout)
        
        # 다크 테마
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #d4d4d4; }
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
            }
            QPushButton:hover { background-color: #1177bb; }
            QLineEdit {
                background-color: #2a2a2a;
                border: 1px solid #3c3c3c;
                color: #d4d4d4;
                padding: 5px;
            }
            QTableWidget {
                background-color: #2a2a2a;
                border: 1px solid #3c3c3c;
                gridline-color: #3c3c3c;
            }
            QTableWidget::item { padding: 5px; }
            QRadioButton { spacing: 10px; }
        """)
    
    def open_json_folder(self):
        """JSON 폴더 열기"""
        json_folder = Path(r"D:\LLM_Dataset\output\Classification Info")
        json_folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(json_folder))
    
    def load_json_file(self):
        """JSON 파일 로드"""
        json_folder = Path(r"D:\LLM_Dataset\output\Classification Info")
        json_folder.mkdir(parents=True, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "분류 JSON 파일 선택",
            str(json_folder),
            "JSON 파일 (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.classification_data = json.load(f)
            
            self.json_path = Path(file_path)
            self.json_path_label.setText(str(self.json_path))
            self.json_path_label.setStyleSheet("QLabel { color: #4a9eff; padding: 5px; }")
            
            # 정보 표시
            self.display_classification_info()
            
            self.run_btn.setEnabled(True)
            self.log_text.clear()
            self.log_text.append(f"✓ JSON 파일 로드 완료: {self.json_path.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"JSON 파일 로드 실패:\n{str(e)}")
    
    def display_classification_info(self):
        """분류 정보 표시"""
        if not self.classification_data:
            return
        
        meta = self.classification_data.get("metadata", {})
        stats = self.classification_data.get("statistics", {}).get("by_class", {})
        
        # 메타데이터
        meta_text = f"""
<b>📁 소스 폴더:</b> {Path(meta.get('source_folder', 'N/A')).name}<br>
<b>🤖 모델:</b> {meta.get('model_name', 'N/A')}<br>
<b>📊 총 이미지:</b> {meta.get('total_images', 0)}개<br>
<b>📅 생성일:</b> {meta.get('created_at', 'N/A')[:19]}
        """
        self.meta_label.setText(meta_text)
        
        # 통계 테이블
        self.stats_table.setRowCount(len(stats))
        for i, (class_name, data) in enumerate(stats.items()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(data.get("count", 0))))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{data.get('percentage', 0):.1f}%"))
            
            # 색상
            if class_name == "Normal":
                color = QColor("#4caf50")
            elif class_name == "Twist":
                color = QColor("#ff9800")
            elif class_name == "Hook":
                color = QColor("#f44336")
            else:
                color = QColor("#4a9eff")
            
            for j in range(3):
                item = self.stats_table.item(i, j)
                if item:
                    item.setForeground(color)
    
    def browse_search_root(self):
        """검색 루트 폴더 변경"""
        folder = QFileDialog.getExistingDirectory(
            self, "검색 루트 폴더 선택",
            str(self.search_root)
        )
        if folder:
            self.search_root = Path(folder)
            self.search_root_edit.setText(str(self.search_root))
    
    def start_moving(self):
        """이동 시작"""
        if not self.classification_data:
            return
        
        # 검색 루트 업데이트
        self.search_root = Path(self.search_root_edit.text())
        
        if not self.search_root.exists():
            QMessageBox.warning(self, "경고", "검색 루트 폴더가 존재하지 않습니다.")
            return
        
        # 확인
        source_folder = Path(self.classification_data["metadata"]["source_folder"]).name
        total = self.classification_data["metadata"]["total_images"]
        mode = "이동" if self.move_radio.isChecked() else "복사"
        
        reply = QMessageBox.question(
            self, "확인",
            f"다음 작업을 진행하시겠습니까?\n\n"
            f"📁 대상 폴더: {source_folder}\n"
            f"📊 이미지 수: {total}개\n"
            f"🔄 작업 모드: {mode}\n\n"
            f"⚠️ 원본 폴더 내에 분류별 서브폴더가 생성됩니다.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # UI 상태 변경
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log_text.append("\n" + "="*50)
        self.log_text.append(f"🚀 작업 시작 ({mode} 모드)")
        
        # 스레드 시작
        self.mover_thread = MoverThread(
            self.classification_data,
            self.search_root,
            move_mode=self.move_radio.isChecked()
        )
        self.mover_thread.progress.connect(self.on_progress)
        self.mover_thread.finished.connect(self.on_finished)
        self.mover_thread.error.connect(self.on_error)
        self.mover_thread.log.connect(self.on_log)
        self.mover_thread.start()
    
    def stop_moving(self):
        """이동 중지"""
        if self.mover_thread:
            self.mover_thread.stop()
    
    def on_progress(self, current, total, filename):
        """진행 상황"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} - {filename}")
    
    def on_log(self, message):
        """로그 메시지"""
        self.log_text.append(message)
        # 스크롤 아래로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_finished(self, success, failed, skipped):
        """완료"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.log_text.append("="*50)
        self.log_text.append(f"✅ 작업 완료!")
        self.log_text.append(f"   • 성공: {success}개")
        self.log_text.append(f"   • 실패: {failed}개")
        self.log_text.append(f"   • 건너뜀: {skipped}개")
        
        QMessageBox.information(
            self, "완료",
            f"작업이 완료되었습니다.\n\n"
            f"✅ 성공: {success}개\n"
            f"❌ 실패: {failed}개\n"
            f"⏭️ 건너뜀: {skipped}개"
        )
    
    def on_error(self, error_msg):
        """에러"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.log_text.append(f"❌ 오류: {error_msg}")
        
        # 자동화 모드면 다음 파일로 계속
        if self.is_automation_running:
            self.automation_results.append({
                "file": self.json_path.name if self.json_path else "Unknown",
                "status": "error",
                "message": error_msg
            })
            self.automation_index += 1
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(500, self.process_next_automation_json)
        else:
            QMessageBox.critical(self, "오류", f"작업 중 오류 발생:\n{error_msg}")
    
    def start_automation(self):
        """자동화 시작 - 여러 JSON 파일 선택"""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QListWidget, QAbstractItemView
        
        json_folder = Path(r"D:\LLM_Dataset\output\Classification Info")
        json_folder.mkdir(parents=True, exist_ok=True)
        
        # 파일 선택 다이얼로그
        dialog = QDialog(self)
        dialog.setWindowTitle("자동화 - JSON 파일 선택")
        dialog.setMinimumSize(700, 500)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #d4d4d4;
            }
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #3c3c3c;
                color: #d4d4d4;
            }
            QListWidget::item:selected {
                background-color: #0e639c;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        # 안내 레이블
        info_label = QLabel("처리할 JSON 파일들을 선택하세요 (Ctrl+클릭으로 다중 선택)")
        info_label.setStyleSheet("QLabel { font-size: 12pt; padding: 10px; }")
        layout.addWidget(info_label)
        
        # 버튼 레이아웃
        btn_layout = QHBoxLayout()
        
        add_files_btn = QPushButton("📄 파일 추가")
        btn_layout.addWidget(add_files_btn)
        
        add_all_btn = QPushButton("📁 폴더 내 전체 추가")
        btn_layout.addWidget(add_all_btn)
        
        btn_layout.addStretch()
        
        clear_btn = QPushButton("🗑 목록 비우기")
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)
        
        # 파일 리스트
        file_list = QListWidget()
        file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(file_list)
        
        # 선택된 파일 수 레이블
        count_label = QLabel("선택된 파일: 0개")
        count_label.setStyleSheet("QLabel { color: #4a9eff; font-weight: bold; }")
        layout.addWidget(count_label)
        
        def add_files():
            files, _ = QFileDialog.getOpenFileNames(
                dialog, "JSON 파일 선택",
                str(json_folder),
                "JSON 파일 (*.json)"
            )
            for f in files:
                # 이미 있는지 확인
                existing = [file_list.item(i).data(Qt.UserRole) for i in range(file_list.count())]
                if f not in existing:
                    item = QListWidgetItem(Path(f).name)
                    item.setData(Qt.UserRole, f)
                    file_list.addItem(item)
            count_label.setText(f"선택된 파일: {file_list.count()}개")
        
        def add_all_from_folder():
            folder = QFileDialog.getExistingDirectory(
                dialog, "JSON 폴더 선택",
                str(json_folder)
            )
            if folder:
                folder_path = Path(folder)
                existing = [file_list.item(i).data(Qt.UserRole) for i in range(file_list.count())]
                for json_file in sorted(folder_path.glob("*.json")):
                    if str(json_file) not in existing:
                        item = QListWidgetItem(json_file.name)
                        item.setData(Qt.UserRole, str(json_file))
                        file_list.addItem(item)
                count_label.setText(f"선택된 파일: {file_list.count()}개")
        
        def clear_files():
            file_list.clear()
            count_label.setText("선택된 파일: 0개")
        
        add_files_btn.clicked.connect(add_files)
        add_all_btn.clicked.connect(add_all_from_folder)
        clear_btn.clicked.connect(clear_files)
        
        # 버튼 박스
        button_box = QDialogButtonBox()
        start_btn = button_box.addButton("▶ 자동화 시작", QDialogButtonBox.AcceptRole)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0a7a0a;
                padding: 10px 30px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #0c9a0c;
            }
        """)
        cancel_btn = button_box.addButton("취소", QDialogButtonBox.RejectRole)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            # 파일 목록 가져오기
            json_files = [file_list.item(i).data(Qt.UserRole) for i in range(file_list.count())]
            
            if not json_files:
                QMessageBox.warning(self, "경고", "처리할 JSON 파일을 추가하세요.")
                return
            
            # 검색 루트 확인
            self.search_root = Path(self.search_root_edit.text())
            if not self.search_root.exists():
                QMessageBox.warning(self, "경고", "검색 루트 폴더가 존재하지 않습니다.")
                return
            
            # 확인
            mode = "이동" if self.move_radio.isChecked() else "복사"
            reply = QMessageBox.question(
                self, "자동화 확인",
                f"다음 작업을 진행하시겠습니까?\n\n"
                f"📄 JSON 파일: {len(json_files)}개\n"
                f"🔄 작업 모드: {mode}\n\n"
                f"⚠️ 각 폴더 내에 분류별 서브폴더가 생성됩니다.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # 자동화 시작
            self.automation_json_files = json_files
            self.automation_index = 0
            self.is_automation_running = True
            self.automation_results = []
            
            # UI 상태 변경
            self.auto_btn.setEnabled(False)
            self.auto_btn.setText(f"⚡ 자동화 중... (0/{len(json_files)})")
            self.run_btn.setEnabled(False)
            self.load_json_btn.setEnabled(False)
            
            self.log_text.clear()
            self.log_text.append(f"🚀 자동화 시작: {len(json_files)}개 파일")
            self.log_text.append("=" * 50)
            
            # 첫 번째 파일 처리 시작
            self.process_next_automation_json()
    
    def process_next_automation_json(self):
        """자동화 - 다음 JSON 파일 처리"""
        if not self.is_automation_running:
            return
        
        if self.automation_index >= len(self.automation_json_files):
            # 모든 파일 처리 완료
            self.finish_automation()
            return
        
        json_file = self.automation_json_files[self.automation_index]
        self.auto_btn.setText(f"⚡ 자동화 중... ({self.automation_index + 1}/{len(self.automation_json_files)})")
        
        self.log_text.append(f"\n📄 [{self.automation_index + 1}/{len(self.automation_json_files)}] {Path(json_file).name}")
        
        try:
            # JSON 파일 로드
            with open(json_file, 'r', encoding='utf-8') as f:
                self.classification_data = json.load(f)
            
            self.json_path = Path(json_file)
            self.json_path_label.setText(str(self.json_path))
            self.json_path_label.setStyleSheet("QLabel { color: #ff6f00; padding: 5px; }")
            
            # 정보 표시
            self.display_classification_info()
            
            # 이동 시작 (확인 없이)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.stop_btn.setEnabled(True)
            
            self.mover_thread = MoverThread(
                self.classification_data,
                self.search_root,
                move_mode=self.move_radio.isChecked()
            )
            self.mover_thread.progress.connect(self.on_progress)
            self.mover_thread.finished.connect(self.on_automation_finished)
            self.mover_thread.error.connect(self.on_error)
            self.mover_thread.log.connect(self.on_log)
            self.mover_thread.start()
            
        except Exception as e:
            self.log_text.append(f"❌ JSON 로드 실패: {e}")
            self.automation_results.append({
                "file": Path(json_file).name,
                "status": "error",
                "message": str(e)
            })
            self.automation_index += 1
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(500, self.process_next_automation_json)
    
    def on_automation_finished(self, success, failed, skipped):
        """자동화 - 개별 파일 처리 완료"""
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        
        # 결과 저장
        self.automation_results.append({
            "file": self.json_path.name if self.json_path else "Unknown",
            "status": "success",
            "success": success,
            "failed": failed,
            "skipped": skipped
        })
        
        self.log_text.append(f"   ✓ 성공: {success}개 | 실패: {failed}개 | 건너뛰: {skipped}개")
        
        # 다음 파일 처리
        self.automation_index += 1
        
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(500, self.process_next_automation_json)
    
    def finish_automation(self):
        """자동화 완료"""
        self.is_automation_running = False
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("⚡ 자동화")
        self.run_btn.setEnabled(True)
        self.load_json_btn.setEnabled(True)
        self.json_path_label.setStyleSheet("QLabel { color: #4a9eff; padding: 5px; }")
        
        # 결과 요약
        total_files = len(self.automation_results)
        success_files = sum(1 for r in self.automation_results if r["status"] == "success")
        error_files = total_files - success_files
        
        total_success = sum(r.get("success", 0) for r in self.automation_results)
        total_failed = sum(r.get("failed", 0) for r in self.automation_results)
        total_skipped = sum(r.get("skipped", 0) for r in self.automation_results)
        
        self.log_text.append("\n" + "=" * 50)
        self.log_text.append("🎉 자동화 완료!")
        self.log_text.append(f"   📄 처리된 파일: {success_files}/{total_files}개")
        self.log_text.append(f"   ✅ 성공: {total_success}개")
        self.log_text.append(f"   ❌ 실패: {total_failed}개")
        self.log_text.append(f"   ⏭️ 건너뛰: {total_skipped}개")
        
        QMessageBox.information(
            self,
            "자동화 완료",
            f"자동화 처리가 완료되었습니다!\n\n"
            f"📄 처리된 파일: {success_files}/{total_files}개\n"
            f"✅ 성공: {total_success}개\n"
            f"❌ 실패: {total_failed}개\n"
            f"⏭️ 건너뛰: {total_skipped}개"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ClassificationMoverGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
