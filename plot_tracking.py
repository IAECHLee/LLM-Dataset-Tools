"""
권취 줄 추적 결과 시각화 스크립트
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r"D:\LLM_Dataset\tracking_results"

def load_csv(csv_path):
    """CSV 파일 로드"""
    return pd.read_csv(csv_path)

def load_json(json_path):
    """JSON 파일 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_tracking_results(csv_path=None):
    """추적 결과 그래프 생성"""
    
    # CSV 파일 자동 찾기
    if csv_path is None:
        output_path = Path(OUTPUT_DIR)
        csv_files = sorted(output_path.glob("tracking_*.csv"))
        if not csv_files:
            print("No tracking CSV files found!")
            return
        csv_path = csv_files[-1]  # 가장 최근 파일
        print(f"Loading: {csv_path}")
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("No tracking data found!")
        return
    
    # 그래프 생성 (2x2 레이아웃)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('권취 줄 위치 추적 결과', fontsize=16, fontweight='bold')
    
    # 1. X, Y 좌표 시계열 그래프
    ax1 = axes[0, 0]
    ax1.plot(df['frame'], df['center_x'], 'b-o', markersize=4, label='X 좌표', alpha=0.7)
    ax1.plot(df['frame'], df['center_y'], 'r-s', markersize=4, label='Y 좌표', alpha=0.7)
    ax1.set_xlabel('프레임 번호')
    ax1.set_ylabel('픽셀 좌표')
    ax1.set_title('중심점 좌표 변화')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. X 좌표만 (변화량 분석)
    ax2 = axes[0, 1]
    ax2.plot(df['frame'], df['center_x'], 'b-o', markersize=5)
    ax2.fill_between(df['frame'], df['center_x'], alpha=0.3)
    ax2.set_xlabel('프레임 번호')
    ax2.set_ylabel('X 좌표 (픽셀)')
    ax2.set_title('X 좌표 변화 (수평 위치)')
    ax2.grid(True, alpha=0.3)
    
    # X 좌표 이동 평균 추가
    if len(df) > 5:
        df['x_ma'] = df['center_x'].rolling(window=5, center=True).mean()
        ax2.plot(df['frame'], df['x_ma'], 'r-', linewidth=2, label='이동평균(5)')
        ax2.legend()
    
    # 3. Y 좌표만 (변화량 분석)
    ax3 = axes[1, 0]
    ax3.plot(df['frame'], df['center_y'], 'r-s', markersize=5)
    ax3.fill_between(df['frame'], df['center_y'], alpha=0.3, color='red')
    ax3.set_xlabel('프레임 번호')
    ax3.set_ylabel('Y 좌표 (픽셀)')
    ax3.set_title('Y 좌표 변화 (수직 위치)')
    ax3.grid(True, alpha=0.3)
    
    # Y 좌표 이동 평균 추가
    if len(df) > 5:
        df['y_ma'] = df['center_y'].rolling(window=5, center=True).mean()
        ax3.plot(df['frame'], df['y_ma'], 'b-', linewidth=2, label='이동평균(5)')
        ax3.legend()
    
    # 4. 2D 궤적 그래프
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['center_x'], df['center_y'], c=df['frame'], 
                          cmap='viridis', s=50, alpha=0.7)
    ax4.plot(df['center_x'], df['center_y'], 'k-', alpha=0.3, linewidth=1)
    
    # 시작점과 끝점 표시
    ax4.scatter(df['center_x'].iloc[0], df['center_y'].iloc[0], 
                c='green', s=200, marker='^', label='시작점', zorder=5)
    ax4.scatter(df['center_x'].iloc[-1], df['center_y'].iloc[-1], 
                c='red', s=200, marker='v', label='끝점', zorder=5)
    
    ax4.set_xlabel('X 좌표 (픽셀)')
    ax4.set_ylabel('Y 좌표 (픽셀)')
    ax4.set_title('2D 궤적 (색상: 프레임 번호)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='프레임')
    
    # Y축 반전 (이미지 좌표계)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    
    # 그래프 저장
    output_path = Path(OUTPUT_DIR)
    plot_filename = csv_path.stem.replace('tracking_', 'plot_') + '.png'
    plot_file = output_path / plot_filename
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Plot: {plot_file}")
    
    # 통계 정보 출력
    print("\n" + "=" * 50)
    print("추적 통계")
    print("=" * 50)
    print(f"총 감지 프레임: {len(df)}")
    print(f"프레임 범위: {df['frame'].min()} ~ {df['frame'].max()}")
    print(f"\nX 좌표:")
    print(f"  - 평균: {df['center_x'].mean():.1f}")
    print(f"  - 표준편차: {df['center_x'].std():.1f}")
    print(f"  - 범위: {df['center_x'].min()} ~ {df['center_x'].max()}")
    print(f"\nY 좌표:")
    print(f"  - 평균: {df['center_y'].mean():.1f}")
    print(f"  - 표준편차: {df['center_y'].std():.1f}")
    print(f"  - 범위: {df['center_y'].min()} ~ {df['center_y'].max()}")
    
    # 이동량 계산
    if len(df) > 1:
        dx = df['center_x'].diff().dropna()
        dy = df['center_y'].diff().dropna()
        print(f"\n프레임간 이동량:")
        print(f"  - X 평균 이동: {dx.mean():.2f} px/frame")
        print(f"  - Y 평균 이동: {dy.mean():.2f} px/frame")
        print(f"  - 총 이동 거리: {np.sqrt((dx**2 + dy**2).sum()):.1f} px")
    
    plt.show()
    
    return df

def plot_comparison(csv_paths):
    """여러 추적 결과 비교"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        label = Path(csv_path).stem
        ax.plot(df['center_x'], df['center_y'], '-o', markersize=3, label=label, alpha=0.7)
    
    ax.set_xlabel('X 좌표 (픽셀)')
    ax.set_ylabel('Y 좌표 (픽셀)')
    ax.set_title('여러 추적 결과 비교')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        plot_tracking_results(csv_path)
    else:
        plot_tracking_results()
