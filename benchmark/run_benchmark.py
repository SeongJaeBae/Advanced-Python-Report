"""
벤치마크: Before vs After 비교 측정
- 실행 시간 (평균 ± 표준편차)
- 메모리 사용량 (peak)
- 이미지 경로 탐색 방식 비교 (list vs generator)

실제 YOLO 학습 없이 구조적 차이만 측정하는 lightweight 벤치마크
"""

import time
import tracemalloc
import statistics
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 벤치마크 대상: 이미지 경로 탐색 방식
# ──────────────────────────────────────────────

def make_fake_image_tree(base_dir: str, n: int):
    """synthetic 이미지 경로 트리 생성 (실제 파일 없이 경로만)."""
    os.makedirs(base_dir, exist_ok=True)
    for i in range(n):
        subdir = os.path.join(base_dir, f"subdir_{i % 10}")
        os.makedirs(subdir, exist_ok=True)
        fpath = os.path.join(subdir, f"img_{i:06d}.jpg")
        if not os.path.exists(fpath):
            open(fpath, "w").close()  # 빈 파일 생성


# Before 방식: list로 전부 수집
def get_image_paths_list(folder):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_list.append(os.path.join(root, file))
    return image_list


# After 방식: generator로 lazy 탐색
def iter_image_paths_gen(folder, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)


def measure(func, *args, repeat=10):
    """repeat 횟수만큼 실행하여 시간과 peak 메모리를 측정."""
    times = []
    memories = []
    for _ in range(repeat):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args)
        # generator는 소비해야 실제 실행됨
        if hasattr(result, "__iter__") and not isinstance(result, list):
            result = list(result)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(t1 - t0)
        memories.append(peak / 1024)  # KB
    return {
        "mean_time_ms": round(statistics.mean(times) * 1000, 4),
        "std_time_ms": round(statistics.stdev(times) * 1000, 4) if len(times) > 1 else 0,
        "mean_memory_kb": round(statistics.mean(memories), 2),
        "peak_memory_kb": round(max(memories), 2),
    }


def run_benchmark(n_images_list: list, repeat: int = 10):
    rows = []

    for n in n_images_list:
        print(f"\n[벤치마크] 이미지 수: {n}장")
        fake_dir = os.path.join(RESULTS_DIR, f"fake_images_{n}")
        make_fake_image_tree(fake_dir, n)

        before = measure(get_image_paths_list, fake_dir, repeat=repeat)
        after = measure(iter_image_paths_gen, fake_dir, repeat=repeat)

        print(f"  Before (list)    시간: {before['mean_time_ms']:.4f}ms ± {before['std_time_ms']:.4f}ms  |  메모리: {before['peak_memory_kb']:.1f}KB")
        print(f"  After  (generator) 시간: {after['mean_time_ms']:.4f}ms ± {after['std_time_ms']:.4f}ms  |  메모리: {after['peak_memory_kb']:.1f}KB")

        speedup = before["mean_time_ms"] / after["mean_time_ms"] if after["mean_time_ms"] > 0 else 0
        mem_reduction = (1 - after["peak_memory_kb"] / before["peak_memory_kb"]) * 100 if before["peak_memory_kb"] > 0 else 0

        print(f"  → 속도: {speedup:.2f}x  |  메모리 절감: {mem_reduction:.1f}%")

        rows.append({
            "n_images": n,
            "before_time_ms": before["mean_time_ms"],
            "before_std_ms": before["std_time_ms"],
            "before_mem_kb": before["peak_memory_kb"],
            "after_time_ms": after["mean_time_ms"],
            "after_std_ms": after["std_time_ms"],
            "after_mem_kb": after["peak_memory_kb"],
            "speedup": round(speedup, 3),
            "mem_reduction_pct": round(mem_reduction, 1),
        })

    # CSV 저장
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n결과 저장 완료: {csv_path}")
    return rows


if __name__ == "__main__":
    import platform
    import sys

    print("=" * 50)
    print("측정 환경")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  반복 횟수: 10회")
    print(f"  Random seed: 없음 (파일 I/O 벤치마크)")
    print("=" * 50)

    run_benchmark(
        n_images_list=[100, 500, 1000, 5000],
        repeat=10,
    )
