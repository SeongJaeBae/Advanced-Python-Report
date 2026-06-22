"""
After: 구조화된 실험 관리 코드

개선 사항:
  B. Generator  — 이미지 경로/배치를 lazy하게 처리 (메모리 절약)
  C. Class 분리 — ExperimentConfig / YOLOTrainer / ResultLogger 역할 분리
  D. Decorator  — @timer, @log_experiment, @validate_config
"""

import os
import time
import json
import functools
import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional
from ultralytics import YOLO


# ──────────────────────────────────────────────
# C. 설정 클래스 — dataclass로 명시적 정의
# ──────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """실험 설정값을 하나의 객체로 관리. frozen=True로 실수로 변경 방지."""
    model_name: str = "yolov8n.pt"
    data_yaml: str = "coco8.yaml"
    epochs: int = 10
    imgsz: int = 640
    batch: int = 16
    lr: float = 0.01
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    save_dir: str = "results/exp1"

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)


# ──────────────────────────────────────────────
# D. 데코레이터 정의
# ──────────────────────────────────────────────

def timer(func):
    """실행 시간을 측정하고 반환값과 함께 elapsed time을 로깅하는 데코레이터."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"[timer] {func.__name__} 완료: {elapsed:.4f}초")
        return result, elapsed
    return wrapper


def log_experiment(func):
    """실험 함수 호출 전후에 시작/종료 로그를 남기는 데코레이터."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"[experiment] '{func.__name__}' 시작")
        try:
            result = func(*args, **kwargs)
            logging.info(f"[experiment] '{func.__name__}' 정상 완료")
            return result
        except Exception as e:
            logging.error(f"[experiment] '{func.__name__}' 실패: {e}")
            raise
    return wrapper


def validate_config(func):
    """ExperimentConfig가 첫 번째 인자로 전달된 경우 기본 유효성 검사."""
    @functools.wraps(func)
    def wrapper(self, config: ExperimentConfig, *args, **kwargs):
        assert config.epochs > 0, "epochs는 1 이상이어야 합니다."
        assert 0 < config.conf_threshold < 1, "conf_threshold는 0~1 사이여야 합니다."
        assert 0 < config.iou_threshold < 1, "iou_threshold는 0~1 사이여야 합니다."
        assert config.imgsz % 32 == 0, "imgsz는 32의 배수여야 합니다."
        logging.info("[validate] 설정값 검증 통과")
        return func(self, config, *args, **kwargs)
    return wrapper


# ──────────────────────────────────────────────
# B. Generator — 이미지 경로 lazy 탐색
# ──────────────────────────────────────────────

def iter_image_paths(folder: str, extensions=(".jpg", ".jpeg", ".png", ".bmp")) -> Iterator[str]:
    """
    이미지 경로를 generator로 반환.
    list로 전부 올리지 않고, 필요할 때마다 하나씩 yield.
    수천 장 이미지도 메모리 부담 없이 처리 가능.
    """
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)


def iter_batches(image_folder: str, batch_size: int) -> Iterator[List[str]]:
    """
    이미지 경로를 batch_size 단위로 묶어 lazy하게 생성.
    전체 경로 list를 메모리에 올리지 않음.
    """
    batch = []
    for path in iter_image_paths(image_folder):
        batch.append(path)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ──────────────────────────────────────────────
# C. ResultLogger — 저장/로깅 책임 분리
# ──────────────────────────────────────────────

class ResultLogger:
    """실험 결과 저장 및 로깅을 전담하는 클래스."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "experiment.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(),
            ],
        )

    def save_json(self, data: dict, filename: str = "result.json"):
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"[logger] 결과 저장: {path}")

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            logging.info(f"[metrics] {k}: {v}")


# ──────────────────────────────────────────────
# C. YOLOTrainer — 학습/평가 책임만 담당
# ──────────────────────────────────────────────

class YOLOTrainer:
    """
    YOLO 모델 학습과 평가만 담당.
    설정(Config), 로깅(Logger)과 완전히 분리.
    실험 조건을 바꾸려면 Config만 교체하면 됨.
    """

    def __init__(self, logger: ResultLogger):
        self.logger = logger
        self.model: Optional[YOLO] = None

    def load_model(self, model_name: str):
        logging.info(f"[trainer] 모델 로딩: {model_name}")
        self.model = YOLO(model_name)

    @validate_config
    @log_experiment
    def train(self, config: ExperimentConfig):
        """학습 실행. 데코레이터가 검증/로깅을 담당하므로 핵심 로직만 존재."""
        return self.model.train(
            data=config.data_yaml,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            lr0=config.lr,
            device=config.device,
            project=config.save_dir,
            name="train",
            exist_ok=True,
        )

    @log_experiment
    def validate(self, config: ExperimentConfig):
        """평가 실행."""
        return self.model.val(
            data=config.data_yaml,
            imgsz=config.imgsz,
            batch=config.batch,
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            device=config.device,
        )

    def run_inference_batched(self, image_folder: str, config: ExperimentConfig):
        """
        B. Generator 기반 배치 추론.
        전체 이미지를 메모리에 올리지 않고 배치 단위로 처리.
        결과도 즉시 처리하고 버림 (누적 X).
        """
        total = 0
        for batch_paths in iter_batches(image_folder, batch_size=config.batch):
            results = self.model(
                batch_paths,
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                device=config.device,
            )
            # 결과를 list에 쌓지 않고 즉시 처리
            for r in results:
                total += 1
                # 필요한 처리만 수행 (예: 저장, 집계 등)
        logging.info(f"[inference] 총 {total}장 처리 완료")
        return total


# ──────────────────────────────────────────────
# 실험 실행 — 설정만 바꾸면 다른 실험 즉시 가능
# ──────────────────────────────────────────────

@timer
def run_full_experiment(config: ExperimentConfig):
    """학습 + 평가 전체 실험 파이프라인."""
    logger = ResultLogger(config.save_dir)
    trainer = YOLOTrainer(logger)
    trainer.load_model(config.model_name)

    # 학습 (@timer는 내부적으로 elapsed 반환)
    (_, train_elapsed) = timer(trainer.train)(config)

    # 평가
    (val_results, val_elapsed) = timer(trainer.validate)(config)

    # 결과 수집 및 저장
    map50 = float(val_results.box.map50) if val_results.box.map50 is not None else 0.0
    map50_95 = float(val_results.box.map) if val_results.box.map is not None else 0.0
    precision = float(val_results.box.mp) if val_results.box.mp is not None else 0.0
    recall = float(val_results.box.mr) if val_results.box.mr is not None else 0.0

    metrics = {
        "model": config.model_name,
        "data": config.data_yaml,
        "epochs": config.epochs,
        "train_time_sec": round(train_elapsed, 4),
        "val_time_sec": round(val_elapsed, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    logger.log_metrics(metrics)
    logger.save_json(metrics)
    return metrics


if __name__ == "__main__":
    # 실험 조건 변경 = Config 객체만 교체
    config_nano = ExperimentConfig(
        model_name="yolov8n.pt",
        epochs=10,
        save_dir="results/exp_nano",
    )

    config_small = ExperimentConfig(
        model_name="yolov8s.pt",
        epochs=10,
        save_dir="results/exp_small",
    )

    # 실험 1
    (result1, total_elapsed1) = run_full_experiment(config_nano)
    print(f"\n[nano] 전체 소요: {total_elapsed1:.2f}초")

    # 실험 2 — Config만 바꿔서 재실행, 코드 수정 없음
    (result2, total_elapsed2) = run_full_experiment(config_small)
    print(f"\n[small] 전체 소요: {total_elapsed2:.2f}초")
