import numpy as np
import mpmath as mp
from numpy.polynomial import Polynomial
from math import log2
from .print import debug_print
from polyEval.algorithm import cal_polyEval
from polyEval.error_bound import cal_bound, ErrBound

''' 기타 필요한 함수들 '''
# 다항함수 평가
def evalP(coeff, x):
    return sum(coeff[i] * x**i for i in range(len(coeff)))

def error_abs(coeff, x, y):
    return abs(evalP(coeff, x) - y)

def error_func(coeff, x, y):
    return evalP(coeff, x) - y

# 구간 [a, b]에서 chebyshev node를 사용하여 m개의 점 샘플링
def sample_points(a: np.float64, b: np.float64, m) -> np.ndarray:
    if m <= 0:
        print("Sample 0 points!!!")
        return np.ndarray([], dtype=float)
    k = np.arange(m)
    x = np.cos(np.pi * (2 * k + 1) / (2 * m))
    return 0.5 * (b + a) + 0.5 * (b - a) * x

def sample_points_multi(m, intervals, print_mode) -> np.ndarray:
    # 구간의 총 길이를 계산
    lengths = np.array([end - start for start, end in intervals])
    total_length = lengths.sum()
    n = len(intervals)
    m_alloc = np.ones(n, dtype=int)
    left = m - n

    if total_length == 0:
        # 총 길이가 0이면 m을 균등하게 분배
        base = m // n
        remainder = m % n
        m_alloc = np.full(n, base, dtype=int)
        if remainder > 0:
            m_alloc[:remainder] += 1
    else:
        # 각 구간의 길이 비율로 m을 분배, 남은 점은 긴 길이부터 추가로 할당
        m_float = left * lengths / total_length
        additional = np.floor(m_float).astype(int)
        m_alloc += additional

        remainder = m - m_alloc.sum()
        if remainder > 0:
            idx_desc = np.argsort(-lengths)
            for idx in idx_desc[:remainder]:
                m_alloc[idx] += 1

    # 구간 별 샘플링
    chunks = []
    for (s, e), num in zip(intervals, m_alloc):
        debug_print(f"sample {num} points in interval[{s}, {e}]", print_mode)
        if num > 0:
            chunks.append(sample_points(s, e, num))

    x = np.concatenate(chunks)
    x.sort()
    return x

def generate_points(start: np.float64, end: np.float64, step=np.float64(1e-8)):
    max_points = 10000  # 최대 허용 포인트 수
    step = np.float64(step)

    while True:
        p = np.arange(start, end + step, step, dtype=np.float64)
        if p.size <= max_points:
            return p
        step *= np.float64(10.0)

# 구간 변경
def slice_interval(approx_mode: str, intervals: list) -> list:
    if approx_mode == "all":
        return intervals
    
    # 기존 구간 intervals 정렬
    new_intervals = []
    for start, end in intervals:
        if start < 0 and end < 0:
            continue
        elif start < 0 and end > 0:
            if abs(start) > abs(end):
                new_intervals.append([0.0, abs(start)])
            else:
                new_intervals.append([0.0, end])
        else:
            new_intervals.append([start, end])

    return new_intervals
            
# 행렬식 구성
def create_matrix(powers: list, x: np.ndarray, evalF) -> tuple[np.ndarray, np.ndarray]:
    A_matrix = []
    y_matrix = []
    for xindex in range(len(x)):
        x_sample = x[xindex]
        rowA = [x_sample ** p for p in powers] + [(-1) ** xindex]
        y = evalF(x_sample)
        A_matrix.append(rowA)
        y_matrix.append(y)

    A_matrix = np.array(A_matrix, dtype=np.float64)
    y_matrix = np.array(y_matrix, dtype=np.float64)
    return A_matrix, y_matrix

# 행렬식 계산 및 계수벡터 연산
def solve_matrix(A: np.ndarray, y: np.ndarray, n: int, powers: list) -> tuple[list, float]:
    # try:
    B = np.linalg.solve(A, y)
    # except np.linalg.LinAlgError:
    #     # print("singular matrix error!")
    #     # print(f"A:\n{A}")
    #     # return solve_matrix_fallback_svd(A, y, n, powers)
    #     raise
    E = B[-1]
    coeff = []
    for k in range(n + 1):
        if k in powers:
            coeff.append(B[0])
            B = B[1:]
        else:
            coeff.append(0.0)
    return coeff, E
    
# def solve_matrix(
#     A: np.ndarray,
#     y: np.ndarray,
#     n: int,
#     powers: list,
#     dps: int = 80,          # 원하는 소수 자릿수(정밀도)
#     use_str: bool = True    # float->mpf 변환 시 str 사용(이진부동소수 오염 완화)
# ) -> tuple[list[np.float64], np.float64]:
#     mp.mp.dps = dps

#     def to_mpf(v):
#         return mp.mpf(str(v)) if use_str else mp.mpf(v)

#     # numpy -> mpmath matrix
#     A_mp = mp.matrix([[to_mpf(A[i, j]) for j in range(A.shape[1])]
#                       for i in range(A.shape[0])])

#     y_arr = np.asarray(y).reshape(-1)
#     y_mp = mp.matrix([to_mpf(v) for v in y_arr])

#     try:
#         B_mp = mp.lu_solve(A_mp, y_mp)
#     except ZeroDivisionError:
#         print("singular matrix error!")
#         print(f"A:\n{A}")
#         return [-1.0], np.float64(-1.0)

#     B_list = [B_mp[i] for i in range(B_mp.rows)]
#     E = np.float64(B_list[-1])

#     coeff = []
#     idx = 0
#     for k in range(n + 1):
#         if k in powers:
#             coeff.append(np.float64(B_list[idx]))
#             idx += 1
#         else:
#             coeff.append(np.float64(0.0))

#     return coeff, E

# def solve_matrix_fallback_svd(
#     A: np.ndarray,
#     y: np.ndarray,
#     n: int,
#     powers: list,
#     dps: int = 200,
#     use_str: bool = True,
#     rcond: float | None = None,     # None이면 자동(정밀도 기반) / 예: 1e-80 같이 직접 지정 가능
#     col_scale: bool = True,         # 열 스케일링(권장)
#     return_float64: bool = True     # 기존과 동일하게 float64 반환
# ) -> tuple[list, float]:
#     """
#     numpy.linalg.solve에서 예외 발생 시 except 블록에서 호출하는 SVD 기반 폴백.
#     - 열 스케일링 + SVD 의사역행렬(pinv)로 해를 구함.
#     - 반환 형식: (coeff: list[float], E: float)  기존과 동일.
#     - 실패 시: [-1], -1
#     """
#     mp.mp.dps = dps

#     def to_mpf(v):
#         return mp.mpf(str(v)) if use_str else mp.mpf(v)

#     A = np.asarray(A)
#     y_arr = np.asarray(y).reshape(-1)

#     # mp matrix 변환
#     m, ncols = A.shape
#     A_mp = mp.matrix([[to_mpf(A[i, j]) for j in range(ncols)] for i in range(m)])
#     y_mp = mp.matrix([to_mpf(v) for v in y_arr])

#     # 열 스케일링: A_scaled[:,j] = A[:,j] / s_j  (s_j = max|col|)
#     # x_j = x_scaled_j / s_j
#     if col_scale:
#         s = []
#         for j in range(ncols):
#             mx = mp.mpf("0")
#             for i in range(m):
#                 v = abs(A_mp[i, j])
#                 if v > mx:
#                     mx = v
#             if mx == 0:
#                 mx = mp.mpf("1")
#             s.append(mx)

#         A_s = mp.matrix(m, ncols)
#         for i in range(m):
#             for j in range(ncols):
#                 A_s[i, j] = A_mp[i, j] / s[j]
#     else:
#         s = [mp.mpf("1") for _ in range(ncols)]
#         A_s = A_mp

#     # SVD 기반 pinv(A_s) * y
#     try:
#         U, S, V = mp.svd(A_s)  # A = U*diag(S)*V^T
#     except Exception:
#         print("singular matrix error!")
#         print(f"A:\n{A}")
#         return [-1], -1

#     # tolerance 설정
#     smax = mp.mpf("0")
#     for sv in S:
#         if sv > smax:
#             smax = sv

#     if smax == 0:
#         print("singular matrix error!")
#         print(f"A:\n{A}")
#         return [-1], -1

#     if rcond is None:
#         # 정밀도 기반 자동 컷오프(경험적): max(m,n)*eps 정도
#         tol = smax * mp.mpf(max(m, ncols)) * mp.eps * mp.mpf("10")
#     else:
#         tol = smax * mp.mpf(str(rcond))

#     # x_scaled = V * diag(1/S_i) * U^T * y (단, S_i > tol)
#     # 먼저 w = U^T * y
#     w = mp.matrix(len(S), 1)
#     for i in range(len(S)):
#         acc = mp.mpf("0")
#         for k in range(m):
#             acc += U[k, i] * y_mp[k]
#         w[i] = acc

#     # w2 = diag(1/S_i) * w
#     w2 = mp.matrix(len(S), 1)
#     for i, sv in enumerate(S):
#         if sv > tol:
#             w2[i] = w[i] / sv
#         else:
#             w2[i] = mp.mpf("0")

#     # x_scaled = V * w2
#     x_scaled = mp.matrix(ncols, 1)
#     for i in range(ncols):
#         acc = mp.mpf("0")
#         for k in range(len(S)):
#             acc += V[i, k] * w2[k]
#         x_scaled[i] = acc

#     # 언스케일: x_j = x_scaled_j / s_j
#     x = mp.matrix(ncols, 1)
#     for j in range(ncols):
#         x[j] = x_scaled[j] / s[j]

#     # 기존 반환 포맷으로 변환
#     B_list = [x[i] for i in range(ncols)]
#     E_val = B_list[-1]

#     coeff = []
#     idx = 0
#     powers_set = set(powers)
#     for k in range(n + 1):
#         if k in powers_set:
#             coeff.append(B_list[idx])
#             idx += 1
#         else:
#             coeff.append(mp.mpf("0"))

#     if return_float64:
#         coeff = [np.float64(c) for c in coeff]
#         E_out = np.float64(E_val)
#     else:
#         E_out = E_val

#     return coeff, E_out

# 지역 극점의 x좌표 계산
def calculate_local_max(coeff, evalF, intervals: list) -> tuple:
    poly = Polynomial(coeff)  # 다항식 객체 생성
    deriv_roots = poly.deriv().roots() # 도함수의 근 계산
    real_roots = deriv_roots[np.isreal(deriv_roots)].real # 실근만 추리기
    
    max_point_x = []
    max_point_y = []
    
    # 각 구간에 속하는 점만 선별
    for start, end in intervals:
        valid_roots = real_roots[(real_roots > start) & (real_roots < end)]
        
        # 구간 시작/끝점, 구간 내 극점들
        candidates = np.unique(np.concatenate(([start, end], valid_roots)))
        
        f_vals = np.array([evalF(x) for x in candidates])
        p_vals = poly(candidates)
        errors = p_vals - f_vals
        
        # 극점 모두 추가
        max_point_x.extend(candidates)
        max_point_y.extend(errors)

    return max_point_x, max_point_y

# mu(x) 계산
def eval_mu(coeff, x: float, eps: float = 0.0) -> int:
        poly = Polynomial(coeff)
        r2 = poly.deriv(2)(x)  # p''(x)
        if r2 < -eps:
            return 1
        if r2 > eps:
            return -1
        return 0 
    
# 논문에서 제시한 극점 선별 알고리즘 구현(Algorithm 3)
def select_max_points(max_point_x: list, max_point_y: list, sample_number: int, coeff: list, evalF, print_mode: str = "normal", E: float | None = None):
    errs = []
    for x, y in zip(max_point_x, max_point_y):
        err = evalP(coeff, x) - evalF(x)
        sgn = 1 if err > 0 else -1
        errs.append((sgn, err, x))
    
    # Step1 - Alternating conditon.
    # 동일한 부호가 발생하는 경우 최대오차만 남기고 제거.
    res_step1 = [errs[0]]
    for i, err_data in enumerate(errs):
        if i == 0:
            continue
        
        compare_data = res_step1[-1]
        if compare_data[0] != err_data[0]:
            res_step1.append(err_data)
        else:
            if abs(compare_data[1]) < abs(err_data[1]):
                res_step1[-1] = err_data
        
    if len(res_step1) < sample_number:
        return [], []  
    
    # Step2 - Maximizing summation of errors
    res_step2 = [(d[0], abs(d[1]), d[2]) for d in res_step1]
    err_sum_neigh = []
    for i in range(0, len(res_step2)-1):
        err_sum_neigh.append((i, res_step2[i][1] + res_step2[i+1][1]))
    err_sum_neigh.sort(key=lambda t: t[1])
    
    while len(res_step2) > sample_number:
        if len(res_step2) == sample_number + 1:
            rm_index = 0 if res_step2[0][1] < res_step2[-1][1] else -1
            del res_step2[rm_index]
        
        elif len(res_step2) >= sample_number + 2:
            err_sum_neigh.append((-1, res_step2[0][1] + res_step2[-1][1]))
            err_sum_neigh.sort(key=lambda t: t[1])
            rm_index = err_sum_neigh[0][0]
            del res_step2[rm_index+1]
            del res_step2[rm_index]
    
    mpx = [data[2] for data in res_step2]
    mpy = [data[0] * data[1] for data in res_step2]
    return mpx, mpy

'''# 논문에서 제시한 극점 선별 알고리즘 구현(Algorithm 3)
def select_max_points(max_point_x: list, max_point_y: list, sample_number: int, coeff: list, evalF, print_mode: str = "normal", E: float | None = None):
    errs = []
    for x, y in zip(max_point_x, max_point_y):
        err = evalP(coeff, x) - evalF(x)
        sgn = 1 if err > 0 else -1
        errs.append((sgn, err, x))
    
    # 후보 데이터 B 연산
    B = []
    for x in max_point_x:
        r = evalP(coeff, x) - evalF(x)
        mu = eval_mu(coeff, x)
        if mu != 0:
            B.append([x, r, mu])
    if len(B) < sample_number:
        return [], []
    
    # Step1 - Alternating conditon.
    # 동일한 부호가 발생하는 경우 최대오차만 남기고 제거.
    res = [B[0]]
    for t in B[1:]:
        prev = res[-1]
        if prev[2] * t[2] == -1:
            res.append(t)
        else:
            if abs(t[1]) > abs(prev[1]):
                res[-1] = t

    if len(res) < sample_number:
        return [], []
    
    # Step2 - Maximizing summation of errors
    while len(res) > sample_number:
        L = len(res)
        
        if L == sample_number + 1:
            if abs(res[0][1]) <= abs(res[-1][1]):
                del res[0]
            else:
                del res[-1]
            continue
        
        if L == sample_number + 2:
            pair_sums = []
            for i in range(L - 1):
                pair_sums.append((abs(res[i][1]) + abs(res[i + 1][1]), i, i + 1))
            pair_sums.append((abs(res[0][1]) + abs(res[-1][1]), 0, L - 1))  # wrap
            pair_sums.sort(key=lambda t: t[0])

            _, i, j = pair_sums[0]
            if i == 0 and j == L - 1:
                del res[-1]
                del res[0]
            else:
                if j > i:
                    del res[j]
                    del res[i]
                else:
                    del res[i]
                    del res[j]
            continue
        
        pair_sums = [(abs(res[i][1]) + abs(res[i + 1][1]), i) for i in range(L - 1)]
        pair_sums.sort(key=lambda t: t[0])
        _, i = pair_sums[0]  # 최소 pair는 (i, i+1)

        if i == 0:
            del res[0]          # t1 포함 -> t1 제거
        elif i == L - 2:
            del res[-1]         # tL 포함 -> tL 제거
        else:
            del res[i + 1]      # 내부 pair -> 둘 다 제거
            del res[i]

    mpx = [t[0] for t in res]
    mpy = [t[1] for t in res]
    return mpx, mpy
'''        
# 종료조건 판단
def decide_exit(errors: list, threshold: float, print_mode: str) -> bool:
    max_err, min_err = max(errors), min(errors)
    debug_print(f"Min/Max error\t\t: {min_err:.5f}, {max_err:.5f}", print_mode)
    if min_err != 0:
        exit_err = (max_err - min_err) / min_err
    else:
        exit_err = 0.0
    debug_print(f"Exit error\t\t: {exit_err}", print_mode)

    if exit_err <= threshold:
        return True
    else:
        return False

# remez 다음을 위한 interval, max_err 계산
def calculate_next_remez(coeff, evalF, eb: ErrBound, intervals):
    try:
        next_interval = []
        max_err = 0
        
        poly = Polynomial(coeff)
        dcmp = cal_polyEval(coeff)
        def calbound(x):
            try:
                bound = cal_bound(eb, x, dcmp)
                return float(bound/eb.scale)
            except Exception as e:
                print(f"x: {x}, bound: {bound}, scale: {eb.scale}")
                return 0.0
        vec_calbound = np.vectorize(calbound, otypes=[float])
        vec_evalF = np.vectorize(evalF, otypes=[float])
        
        for start, end in intervals:
            if start == end:
                points = np.array([start])
            else:
                points = generate_points(start, end)
            
            errors = vec_calbound(points)
            p_vals = poly(points)
            f_vals = vec_evalF(points)
            
            p_vals1 = p_vals + errors
            p_vals2 = p_vals - errors          

            err_vals = np.concatenate((np.abs(f_vals - p_vals1), np.abs(f_vals - p_vals2)))
            vals = np.concatenate((p_vals1, p_vals2))
            next_interval.append([np.min(vals), np.max(vals)])
            
            # 최대오차
            max_err = max(np.max(err_vals), max_err)
        
        # 정렬 - 구간이 긴 쪽이 다음 근사에서 0으로 수렴해야함
        if (next_interval[0][1]-next_interval[0][0]) < (next_interval[1][1]-next_interval[1][0]):
            next_interval.reverse()

        return max_err, next_interval
    except Exception as e:
        print(e)
        return -99, [[0, 1], [1, 0]]




