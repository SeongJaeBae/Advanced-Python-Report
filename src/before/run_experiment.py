"""
Before: 문제가 있는 실험 코드
- 설정값 하드코딩
- 학습/평가/로깅/저장이 하나의 함수에 혼재
- 실험 조건 바꾸려면 코드 직접 수정 필요
- 실행 시간 측정 없음
- 결과를 list에 전부 쌓아서 메모리 낭비
- 같은 경로 탐색 로직 중복
"""

import os
import time
import json
from ultralytics import YOLO


def run_experiment():
    # 설정값 하드코딩 — 실험 조건 바꾸려면 여기 직접 수정해야 함
    model_name = "yolov8n.pt"
    data_yaml = "coco8.yaml"
    epochs = 10
    imgsz = 640
    batch = 16
    lr = 0.01
    save_dir = "results/exp1"
    log_file = "results/exp1/log.txt"
    conf_threshold = 0.25
    iou_threshold = 0.45
    device = "cpu"

    os.makedirs(save_dir, exist_ok=True)

    # 로그 시작 — 핵심 로직과 섞여있음
    with open(log_file, "w") as f:
        f.write(f"실험 시작\n")
        f.write(f"model: {model_name}\n")
        f.write(f"data: {data_yaml}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"imgsz: {imgsz}\n")
        f.write(f"batch: {batch}\n")
        f.write(f"lr: {lr}\n")

    print("모델 로딩 중...")
    model = YOLO(model_name)

    # 학습 시작 시간 측정 — 매번 직접 작성
    train_start = time.time()

    print("학습 시작...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        device=device,
        project=save_dir,
        name="train",
        exist_ok=True,
    )
    train_end = time.time()
    train_elapsed = train_end - train_start

    print(f"학습 완료. 소요 시간: {train_elapsed:.2f}초")

    with open(log_file, "a") as f:
        f.write(f"학습 소요 시간: {train_elapsed:.2f}초\n")

    # 평가 — 학습 함수 안에서 바로 이어서 실행
    print("평가 시작...")
    val_start = time.time()

    val_results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
    )
    val_end = time.time()
    val_elapsed = val_end - val_start

    print(f"평가 완료. 소요 시간: {val_elapsed:.2f}초")

    # 결과 수집 — 모든 결과를 list에 한꺼번에 쌓음
    all_results = []
    
    map50 = float(val_results.box.map50) if val_results.box.map50 is not None else 0.0
    map50_95 = float(val_results.box.map) if val_results.box.map is not None else 0.0
    precision = float(val_results.box.mp) if val_results.box.mp is not None else 0.0
    recall = float(val_results.box.mr) if val_results.box.mr is not None else 0.0

    all_results.append({
        "model": model_name,
        "data": data_yaml,
        "epochs": epochs,
        "train_time": train_elapsed,
        "val_time": val_elapsed,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
    })

    # 저장 — 역시 같은 함수 안에서
    result_path = os.path.join(save_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(log_file, "a") as f:
        f.write(f"평가 소요 시간: {val_elapsed:.2f}초\n")
        f.write(f"mAP50: {map50:.4f}\n")
        f.write(f"mAP50-95: {map50_95:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write("실험 완료\n")

    print(f"결과 저장 완료: {result_path}")
    print(f"mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")

    # 두 번째 실험 — 설정 바꾸려면 위에서부터 다시 수정해야 함
    # 예: model_name = "yolov8s.pt" 로 바꾸고 싶으면 위로 올라가서 수정
    # 결과 비교도 수동으로 해야 함


def get_image_paths(folder):
    """이미지 경로 수집 — 결과를 list로 전부 반환 (메모리 비효율)"""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_list.append(os.path.join(root, file))
    return image_list  # 수천 장이어도 전부 메모리에 올림


def run_inference(model_path, image_folder):
    """추론 실행 — 로깅/저장/에러 처리 없이 단순 실행"""
    # 경로 탐색 로직이 get_image_paths와 중복
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    model = YOLO(model_path)
    all_preds = []

    infer_start = time.time()
    for path in image_paths:
        result = model(path, conf=0.25, iou=0.45)  # conf, iou 또 하드코딩
        all_preds.append(result)  # 전체 결과를 메모리에 누적

    infer_end = time.time()
    print(f"추론 완료: {infer_end - infer_start:.2f}초, {len(all_preds)}장 처리")
    return all_preds


if __name__ == "__main__":
    run_experiment()
