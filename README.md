# Layer Sampler - 이미지 샘플링 도구

권선 이미지를 층별로 분류하고 샘플링하는 PyQt5 기반 GUI 프로그램

## 📁 프로그램 구성

### 핵심 파일
```
layer_sampler_gui.py    # GUI 메인 프로그램 (1500+ 라인)
layer_sampler.py        # 코어 샘플링 로직 (380 라인)
layers.json             # 레이어 정의 파일 (18-19개 층 정보)
sampling_history.json   # 샘플링 히스토리 (자동 생성)
```

### 테스트/유틸리티
```
test_sampling.py        # 샘플링 로직 테스트
test_random_seed.py     # 랜덤 시드 검증
```

---

## 🎯 주요 기능

### 1. **전체 샘플링** (파란색 버튼)
- **K:\LLM Image_Storage** 폴더에서 키워드 검색
- 각 폴더의 이미지를 19개 레이어로 분류
- Normal(정상) / Defect(불량) 자동 구분
- 레이어별로 지정된 비율(기본 5%) 랜덤 샘플링
- 폴더마다 다른 시드 사용 → 최대 다양성

**출력 구조:**
```
output/
├── Normal Layer/
│   ├── Layer_01/
│   ├── Layer_02/
│   └── ...
├── Defect Layer/
│   ├── Layer_01/
│   ├── Layer_02/
│   └── ...
├── manifest.json       # 복사 이력
└── stats.json          # 통계 정보
```

### 2. **부분 샘플링** (보라색 버튼)
- 특정 레이어 또는 카테고리만 추가 샘플링
- **중복 방지**: 이미 샘플링된 이미지 자동 제외
- Defect가 부족할 때 유용

**설정 옵션:**
- 레이어: 전체 / Layer 01-19
- 카테고리: Normal+Defect / Normal만 / Defect만
- 샘플링 비율: 0.1% ~ 100%

### 3. **듀얼 이미지 뷰어**
- Normal / Defect 이미지 좌우 동시 표시
- 레이어별 이미지 개수 실시간 표시
- 이미지 네비게이션:
  - **Space**: 다음 이미지
  - **Backspace**: 이전 이미지
  - 버튼: ◀ 이전 / 다음 ▶

### 4. **이미지 분류 수정**
- **← Normal로 이동**: Defect → Normal
- **Defect로 이동 →**: Normal → Defect
- 이동 후 자동으로 다음 이미지 표시
- 개수 자동 업데이트

### 5. **랜덤 삭제** (노란색 버튼)
- 목표 개수 입력
- 현재 개수에서 목표 개수까지 랜덤 삭제
- Normal / Defect 독립적으로 처리
- 레이어별 정밀 조정 가능

### 6. **샘플링 히스토리 관리**
- **자동 추적**: `sampling_history.json`에 선택된 이미지 저장
- **중복 방지**: 부분 샘플링 시 자동으로 이미 선택된 이미지 제외
- **리스트 클린** (빨간색): 히스토리 초기화 → 새 프로젝트 시작
- **히스토리 보기**: 현재 샘플링 통계 확인

---

## 🚀 사용법

### 기본 워크플로우

```bash
# 1. 환경 활성화 및 실행
conda activate dl_test
python layer_sampler_gui.py
```

#### 시나리오 1: 전체 샘플링
1. 루트 디렉토리: `K:\LLM Image_Storage`
2. 출력 디렉토리: `D:\LLM_Dataset\output`
3. 레이어 파일: `layers.json`
4. 샘플링 비율: 5%
5. 검색 키워드: `A line, 2025-07-27`
6. **"샘플링 시작"** 클릭 → 자동으로 `sampling_history.json` 생성

#### 시나리오 2: Defect 추가 샘플링
1. **"부분 샘플링"** 클릭
2. 레이어: Layer 05 (또는 전체)
3. 카테고리: **Defect만**
4. 샘플링 비율: 10%
5. 확인 → 이미 선택된 이미지는 자동 제외

#### 시나리오 3: 이미지 분류 수정
1. **"이미지 뷰어 열기"**
2. 레이어 선택 (1-19)
3. 이미지 확인 (Space/Backspace로 이동)
4. 잘못 분류된 이미지 발견 시:
   - Defect 이미지 선택 → **"← Normal로 이동"**
   - Normal 이미지 선택 → **"Defect로 이동 →"**

#### 시나리오 4: 개수 조정
1. 레이어 선택
2. Normal 개수 확인: 150개
3. 목표: 100개 입력
4. **"Normal 랜덤 삭제"** → 50개 랜덤 삭제

#### 시나리오 5: 새 프로젝트 시작
1. **"리스트 클린"** 클릭
2. 확인 → `sampling_history.json` 삭제
3. 처음부터 샘플링 가능

---

## 📊 코어 로직 (`layer_sampler.py`)

### 핵심 함수

```python
def find_target_folders(root, keywords)
    # 키워드로 폴더 검색 (A line, 2025-07-27)

def process_folder_with_layers(folder, layers, ratio, seed, exclude_images)
    # 폴더 내 이미지 샘플링
    # - 레이어별 분류
    # - 중복 이미지 제외
    # - 랜덤 샘플링

def assign_layers(images, layers)
    # 이미지 인덱스로 레이어 자동 매핑
```

### 샘플링 알고리즘

1. **폴더 발견**: 키워드 매칭 (`A line`, `2025-07-27`)
2. **카테고리 판단**: 폴더명에서 `(정상)` 또는 `(불량)` 확인
3. **레이어 분류**: 이미지 인덱스 → 레이어 매핑 (layers.json 기준)
4. **중복 제거**: `exclude_images` 세트와 비교
5. **랜덤 샘플링**: 레이어별 `ratio` 비율만큼 선택
6. **시드 생성**: `base_seed + hash(folder_name) % 10000 + (folder_idx * 1000)`
   - 폴더마다 완전히 다른 이미지 선택 보장

---

## 🎨 GUI 구조 (`layer_sampler_gui.py`)

### 주요 클래스

#### 1. `SamplerThread` (QThread)
- 백그라운드 샘플링 처리
- 진행 상황 실시간 업데이트
- 히스토리 자동 관리

#### 2. `ImageViewer` (QWidget)
- 이미지 표시 위젯
- 파일 정보 (크기, 용량)
- 자동 스케일링

#### 3. `LayerSamplerGUI` (QMainWindow)
- 메인 윈도우
- 설정 패널 (왼쪽)
- 로그 & 뷰어 탭 (오른쪽)

#### 4. `PartialSamplingDialog` (QDialog)
- 부분 샘플링 설정 다이얼로그
- 레이어/카테고리/비율 선택

### 레이아웃

```
┌────────────────────────────────────────────────────────────┐
│  Layer Sampler - 이미지 샘플링 도구                        │
├───────────────┬────────────────────────────────────────────┤
│  [설정 패널]  │  [탭: 실행 로그 / 듀얼 이미지 뷰어]        │
│               │                                            │
│ • 루트 디렉토리│  ┌─────────────────┬─────────────────┐    │
│ • 레이어 파일 │  │  Normal Layer   │  Defect Layer   │    │
│ • 출력 디렉토리│  ├─────────────────┼─────────────────┤    │
│ • 샘플링 비율 │  │  [이미지 리스트]│  [이미지 리스트]│    │
│ • 검색 키워드 │  │  [뷰어]         │  [뷰어]         │    │
│               │  │  [◀이전][다음▶] │  [◀이전][다음▶] │    │
│ [샘플링 시작] │  └─────────────────┴─────────────────┘    │
│ [뷰어 열기]   │  [← Normal로] [Defect로 →]                │
│ [부분 샘플링] │  레이어: [1-19] Normal: 150개 목표: 100개 │
│ [리스트 클린] │                                            │
│ [히스토리보기]│                                            │
└───────────────┴────────────────────────────────────────────┘
```

---

## 📝 데이터 파일

### `layers.json`
```json
{
  "A line-2025-07-27_14-31-07": [
    {"layer": 1, "start": 23, "end": 153, "count": 130},
    {"layer": 2, "start": 154, "end": 284, "count": 130},
    ...
    {"layer": 18, "start": 2743, "end": 2868, "count": 125}
  ]
}
```

### `sampling_history.json` (자동 생성)
```json
{
  "A line-2025-07-27_07-04-47(정상)": [
    "A_000119.jpg",
    "A_000106.jpg",
    ...
  ],
  "A line-2025-07-27_07-09-26(불량)": [
    "A_000150.jpg",
    ...
  ]
}
```

### `manifest.json` (자동 생성)
```json
[
  {
    "source": "K:\\LLM Image_Storage\\...",
    "destination": "D:\\LLM_Dataset\\output\\Normal Layer\\Layer_01\\...",
    "folder": "A line-2025-07-27_07-04-47(정상)",
    "category": "normal",
    "layer": 1,
    "filename": "A_000119.jpg"
  }
]
```

---

## ⚙️ 기술 스택

- **Python**: 3.11.13
- **GUI**: PyQt5 5.15.11
- **환경**: Conda (dl_test)
- **OS**: Windows

## 🔧 주요 알고리즘

### 1. 다양성 보장 랜덤 시드
```python
folder_seed = base_seed + hash(folder.name) % 10000 + (folder_idx * 1000)
```
→ 각 폴더마다 완전히 다른 이미지 선택

### 2. 중복 방지
```python
images = [img for img in images if img.name not in exclude_images]
```
→ 부분 샘플링 시 이미 선택된 이미지 자동 제외

### 3. 레이어 자동 매핑
```python
for idx, img in enumerate(images):
    for layer_id, (start, end, count) in enumerate(layers, 1):
        if start <= idx + 1 <= end:
            assign to layer_id
```
→ 이미지 순서로 레이어 자동 결정

---

## 📌 핵심 특징

✅ **중복 방지**: 샘플링 히스토리로 같은 이미지 재선택 방지  
✅ **최대 다양성**: 폴더별 고유 시드로 이미지 다양성 보장  
✅ **유연성**: 전체/부분 샘플링, 레이어별 조정  
✅ **직관적 UI**: 듀얼 뷰어, 키보드 단축키, 실시간 개수 표시  
✅ **안전성**: 확인 메시지, 복구 불가 경고, 상세 로그  

---

## 🎓 사용 팁

1. **첫 샘플링**: 낮은 비율(1-2%)로 시작 → 필요시 부분 샘플링 추가
2. **Defect 부족**: 부분 샘플링으로 Defect만 비율 높여서 추가
3. **분류 수정**: Space/Backspace로 빠르게 검토 → 잘못된 것만 이동
4. **개수 조정**: 랜덤 삭제로 레이어별 균형 맞추기
5. **새 프로젝트**: 리스트 클린으로 히스토리 초기화

---

## 📦 의존성

```bash
conda install -c conda-forge pyqt=5.15.11
```

## 🚦 실행

```bash
conda activate dl_test
python layer_sampler_gui.py
```

---

**개발 일자**: 2025년 11월  
**버전**: 1.0
