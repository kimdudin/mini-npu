"""Mini NPU 시뮬레이터 - MAC 연산 기반 Cross / X 패턴 판별기 (표준 라이브러리만 사용)"""
import json
import os
import re
import time

EPSILON = 1e-9
REPEAT = 10
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")


# ---------------------------------------------------------------- 라벨 정규화
def normalize_label(raw):
    """'+', 'cross' -> 'Cross' / 'x' -> 'X'. 알 수 없으면 None."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if key in ("+", "cross"):
        return "Cross"
    if key == "x":
        return "X"
    return None


# ---------------------------------------------------------------- 패턴 생성기 (보너스)
def generate_cross(n):
    c = n // 2
    return [[1.0 if (i == c or j == c) else 0.0 for j in range(n)] for i in range(n)]


def generate_x(n):
    return [[1.0 if (i == j or i + j == n - 1) else 0.0 for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------- MAC 연산
def mac(pattern, filt):
    """위치별 곱셈 후 누적 합산. 반복문으로 직접 구현."""
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


def mac_flat(pattern, filt):
    """보너스: 1차원 배열(길이 N^2) 버전"""
    total = 0.0
    for k in range(len(pattern)):
        total += pattern[k] * filt[k]
    return total


def is_tie(a, b):
    return abs(a - b) < EPSILON


def judge(score_a, score_b, label_a, label_b):
    if is_tie(score_a, score_b):
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def measure_ms(n, func=mac, repeat=REPEAT):
    """크기 n에 대한 MAC 평균 시간(ms). 연산 함수 호출 구간만 측정."""
    p, f = generate_cross(n), generate_x(n)
    if func is mac_flat:
        p = [v for row in p for v in row]
        f = [v for row in f for v in row]
    total = 0.0
    for _ in range(repeat):
        s = time.perf_counter()
        func(p, f)
        total += time.perf_counter() - s
    return total / repeat * 1000


def print_perf_table(sizes):
    print("\n[성능 분석] (각 크기 %d회 평균)" % REPEAT)
    print("%-10s %-16s %-12s" % ("크기(NxN)", "평균 시간(ms)", "연산 횟수(N²)"))
    print("-" * 42)
    for n in sizes:
        print("%-10s %-16.6f %-12d" % ("%dx%d" % (n, n), measure_ms(n), n * n))


# ---------------------------------------------------------------- 모드 1
def read_matrix(name, n=3):
    print("\n%s (%d줄, 줄마다 숫자 %d개를 공백으로 구분)" % (name, n, n))
    while True:
        rows = []
        try:
            for i in range(n):
                line = input("  %d행: " % (i + 1)).split()
                if len(line) != n:
                    raise ValueError("size")
                rows.append([float(x) for x in line])
            return rows
        except ValueError:
            print("입력 형식 오류: 각 줄에 %d개의 숫자를 공백으로 구분해 입력하세요." % n)
            print("처음부터 다시 입력합니다.")


def mode_user():
    print("\n=== 모드 1: 사용자 입력 (3x3) ===")
    a = read_matrix("필터 A")
    b = read_matrix("필터 B")
    print("\n필터 A, B 저장 완료.")
    p = read_matrix("패턴")
    print("패턴 저장 완료.")

    s = time.perf_counter()
    sa = mac(p, a)
    sb = mac(p, b)
    ms = (time.perf_counter() - s) * 1000
    result = judge(sa, sb, "A", "B")
    print("\nA 점수: %s\nB 점수: %s\n연산 시간: %.6f ms" % (sa, sb, ms))
    print("판정: %s" % ("판정 불가 (동점)" if result == "UNDECIDED" else result))
    print_perf_table([3])


# ---------------------------------------------------------------- 모드 2
def validate_matrix(m, n):
    """올바르면 None, 아니면 사유 문자열."""
    if not isinstance(m, list) or len(m) != n:
        return "행 수 불일치(기대 %d)" % n
    for i, row in enumerate(m):
        if not isinstance(row, list) or len(row) != n:
            return "%d행 열 수 불일치(기대 %d)" % (i + 1, n)
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return "%d행에 숫자가 아닌 값" % (i + 1)
    return None


def load_filters(raw_filters, failures_out):
    """{size: {'Cross': m, 'X': m}} 반환. 문제 있는 필터는 제외하고 사유 기록."""
    result = {}
    for key, group in (raw_filters or {}).items():
        m = re.fullmatch(r"size_(\d+)", key)
        if not m or not isinstance(group, dict):
            failures_out.append(("filters/" + key, "필터 키/스키마 오류"))
            continue
        n = int(m.group(1))
        std = {}
        for raw_label, matrix in group.items():
            label = normalize_label(raw_label)
            if label is None:
                failures_out.append(("filters/%s/%s" % (key, raw_label), "알 수 없는 필터 라벨"))
                continue
            err = validate_matrix(matrix, n)
            if err:
                failures_out.append(("filters/%s/%s" % (key, raw_label), "필터 " + err))
                continue
            std[label] = matrix
        result[n] = std
    return result


def run_case(case_id, item, filters):
    """(PASS/FAIL, 상세, 사유) 반환. 예외가 밖으로 나가지 않는다."""
    m = re.fullmatch(r"size_(\d+)_\w+", case_id)
    if not m:
        return False, None, "패턴 키 형식 오류"
    n = int(m.group(1))
    if not isinstance(item, dict) or "input" not in item or "expected" not in item:
        return False, None, "스키마 오류(input/expected 누락)"
    expected = normalize_label(item["expected"])
    if expected is None:
        return False, None, "expected 라벨 오류(%r)" % (item["expected"],)
    if n not in filters:
        return False, None, "size_%d 필터 없음" % n
    if "Cross" not in filters[n] or "X" not in filters[n]:
        return False, None, "size_%d 필터에 Cross/X 중 누락" % n
    err = validate_matrix(item["input"], n)
    if err:
        return False, None, "패턴 크기 불일치: " + err
    sc = mac(item["input"], filters[n]["Cross"])
    sx = mac(item["input"], filters[n]["X"])
    verdict = judge(sc, sx, "Cross", "X")
    detail = (sc, sx, verdict, expected)
    if verdict == expected:
        return True, detail, ""
    return False, detail, "판정 %s != expected %s" % (verdict, expected)


def mode_json():
    print("\n=== 모드 2: data.json 분석 ===")
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print("data.json 로드 실패: %s" % e)
        return

    failures = []
    filters = load_filters(data.get("filters"), failures)
    print("로드된 필터 크기: %s" % sorted(filters))
    patterns = data.get("patterns") or {}
    total = passed = 0
    pre_fail = len(failures)  # 필터 단계 문제도 실패 목록에 포함

    for case_id, item in patterns.items():
        total += 1
        ok, detail, reason = run_case(case_id, item, filters)
        if detail:
            sc, sx, verdict, exp = detail
            print("%-12s Cross=%-8s X=%-8s 판정=%-9s expected=%-6s %s"
                  % (case_id, sc, sx, verdict, exp, "PASS" if ok else "FAIL"))
        else:
            print("%-12s FAIL (%s)" % (case_id, reason))
        if ok:
            passed += 1
        else:
            failures.append((case_id, reason))

    print_perf_table([3, 5, 13, 25])
    print("\n[결과 요약] 전체 %d / 통과 %d / 실패 %d" % (total, passed, total - passed))
    if failures:
        print("실패/문제 목록:")
        for cid, why in failures:
            print("  - %s: %s" % (cid, why))
    if pre_fail:
        print("(위 목록에는 필터 단계 문제 %d건 포함)" % pre_fail)

# ---------------------------------------------------------------- 보너스 1: 메모리 접근 최적화 (2차원 vs 1차원)
def flatten(m):
    """n x n 2차원 배열 -> 길이 n*n 1차원 배열"""
    return [v for row in m for v in row]


def time_avg_ms(func, a, b, repeat=REPEAT):
    """같은 입력으로 func(a, b)를 repeat회 호출한 평균 시간(ms). 연산 구간만 측정."""
    func(a, b)  # 워밍업 1회 (첫 호출의 준비 시간이 결과를 흔들지 않도록, 측정에는 포함하지 않음)
    total = 0.0
    for _ in range(repeat):
        s = time.perf_counter()
        func(a, b)
        total += time.perf_counter() - s
    return total / repeat * 1000


def mode_compare():
    print("\n=== 보너스 1: 최적화 전/후 성능 비교 (2차원 mac vs 1차원 mac_flat) ===")
    print("동일 입력(Cross 패턴 x X 필터), 동일 반복 횟수(%d회)로 측정합니다." % REPEAT)
    print("%-10s %-14s %-14s %-10s %-10s" % ("크기(NxN)", "2차원(ms)", "1차원(ms)", "속도 비", "점수 일치"))
    print("-" * 62)
    for n in (3, 5, 13, 25):
        p, f = generate_cross(n), generate_x(n)
        pf, ff = flatten(p), flatten(f)
        same = is_tie(mac(p, f), mac_flat(pf, ff))
        t2 = time_avg_ms(mac, p, f)
        t1 = time_avg_ms(mac_flat, pf, ff)
        ratio = t2 / t1 if t1 > 0 else 0.0
        print("%-10s %-14.6f %-14.6f %-10s %-10s"
              % ("%dx%d" % (n, n), t2, t1, "%.2fx" % ratio, "O" if same else "X"))
    print("속도 비 = 2차원 시간 / 1차원 시간 (1보다 크면 1차원이 더 빠름)")


# ---------------------------------------------------------------- 보너스 2: 패턴 생성기
def print_pattern(name, m):
    print("%s:" % name)
    for row in m:
        print("  " + " ".join("#" if v else "." for v in row))


def mode_generator():
    print("\n=== 보너스 2: 패턴 생성기 ===")
    while True:
        raw = input("크기 N 입력 (3 이상의 정수): ").strip()
        if raw.isdigit() and int(raw) >= 3:
            n = int(raw)
            break
        print("입력 오류: 3 이상의 정수를 입력하세요.")
    if n % 2 == 0:
        print("(참고: 짝수 N 은 십자가 중심이 정확히 한가운데가 아니라 한 칸 치우칩니다.)")

    cross, x = generate_cross(n), generate_x(n)
    print_pattern("Cross %dx%d" % (n, n), cross)
    print_pattern("X %dx%d" % (n, n), x)

    # 생성된 패턴을 그대로 필터/입력으로 재활용해 판정
    print("\n생성된 패턴을 입력으로 판정합니다 (필터 = 생성된 Cross / X).")
    for label, pattern in (("Cross", cross), ("X", x)):
        s = time.perf_counter()
        sc, sx = mac(pattern, cross), mac(pattern, x)
        ms = (time.perf_counter() - s) * 1000
        print("입력=%-5s Cross 점수=%-6s X 점수=%-6s 판정=%-9s (%.6f ms)"
              % (label, sc, sx, judge(sc, sx, "Cross", "X"), ms))
    print_perf_table([n])


# ---------------------------------------------------------------- 메인
def main():
    while True:
        print("\n=== Mini NPU 시뮬레이터 ===")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. [보너스] 최적화 전/후 성능 비교 (2차원 vs 1차원)")
        print("4. [보너스] 패턴 생성기 (N 입력)")
        print("0. 종료")
        c = input("선택: ").strip()
        if c == "1":
            mode_user()
        elif c == "2":
            mode_json()
        elif c == "3":
            mode_compare()
        elif c == "4":
            mode_generator()
        elif c == "0":
            break
        else:
            print("0 ~ 4 중에서 선택하세요.")


if __name__ == "__main__":
    main()