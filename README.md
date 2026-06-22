# YOLOv8 실험 관리 코드 최적화

## 프로젝트 개요

ultralytics YOLOv8을 사용하는 객체 탐지 실험 관리 코드를, 수업에서 학습한 Python 최적화 기법을 적용하여 구조적으로 개선한 프로젝트입니다.

## 개선 내용 요약

| 항목 | Before | After |
|------|--------|-------|
| 설정 관리 | 함수 내 하드코딩 | `ExperimentConfig` dataclass |
| 학습/평가/로깅 | 하나의 함수에 혼재 | `YOLOTrainer` / `ResultLogger` 분리 |
| 이미지 경로 탐색 | `list`로 전부 수집 | `generator`로 lazy 처리 |
| 실행 시간 측정 | 매번 직접 작성 | `@timer` 데코레이터 |
| 로깅 | `print` + 파일 직접 작성 | `@log_experiment` 데코레이터 |
| 설정 검증 | 없음 | `@validate_config` 데코레이터 |
| 실험 반복 | 코드 직접 수정 | Config 객체만 교체 |

## 디렉토리 구조

```
project/
├── README.md
├── requirements.txt
├── src/
│   ├── before/
│   │   └── run_experiment.py   # 최적화 전 코드
│   └── after/
│       └── run_experiment.py   # 최적화 후 코드
├── benchmark/
│   └── run_benchmark.py        # 성능 측정 코드
├── results/
│   └── benchmark_results.csv   # 벤치마크 결과
└── report/
    └── report.pdf
```

## 실행 방법

### 환경 설치

```bash
pip install -r requirements.txt
```

### Before 실행

```bash
python src/before/run_experiment.py
```

### After 실행

```bash
python src/after/run_experiment.py
```

### 벤치마크 실행

```bash
python benchmark/run_benchmark.py
```

결과는 `results/benchmark_results.csv`에 저장됩니다.

## 측정 환경

- Python 3.10+
- ultralytics >= 8.0
- 반복 횟수: 10회 (평균 ± 표준편차)
- 입력 크기: 100 / 500 / 1000 / 5000장

## 적용한 수업 개념

- **B. Generator / Lazy Evaluation**: 이미지 경로 탐색 및 배치 생성을 generator로 전환
- **C. Class 설계 / 단일책임원칙**: `ExperimentConfig`, `YOLOTrainer`, `ResultLogger`로 역할 분리
- **D. Decorator**: `@timer`, `@log_experiment`, `@validate_config` 구현
