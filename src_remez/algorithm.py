import numpy as np
from .basic import *
from .basic_multi import find_best_combination
from .print import *
from .print_plot import *
from polyEval.error_bound import ErrBound

# remez algorithm
def remez_algorithm(n: int | list, intervals: list, evalF, approx_mode: str | None, eb: ErrBound, print_mode: str="normal", clni: bool=True) -> tuple | list:  
    err_return = ([-1], 99, []) if clni else [-1]
    if len(intervals) == 0:
        return err_return

    if type(n) == int:           
        # 1: approx_mode 기반 설정
        if approx_mode in ["odd", "even"]:
            powers = [i for i in range(n % 2, n + 1, 2)]
        elif approx_mode == "all":
            powers = [i for i in range(n+1)]    
        sample_number = len(powers) + 1
            
    # list로 주어지는 경우 - 제외할 계수가 0인 binary list 형태.
    elif type(n) == list:
        # 예상되는 다항식이 짝수/홀수인지 확인하고 approx_mode 설정.
        k = len(n)
        flag_even = any(n[i] == 1 for i in range(0, k, 2))
        flag_odd  = any(n[i] == 1 for i in range(1, k, 2))
        if flag_even and not flag_odd:
            approx_mode = "even"
            # sample_number = int(k / 2 + 2)
            # powers = [i for i in range(k % 2, k + 1, 2)]
        elif flag_odd and not flag_even:
            approx_mode = "odd"
            # sample_number = int((k + 1) / 2) + 1
            # powers = [i for i in range(k % 2, k + 1, 2)]
        else:
            approx_mode = "all"
            
        powers = [i for i, v in enumerate(n) if v == 1]
        sample_number = len(powers) + 1
        n = k - 1
        
    intervals = slice_interval(approx_mode, intervals)
    x_samples = sample_points_multi(sample_number, intervals, print_mode)

    for _ in range(50):
        A_matrix, y_matrix = create_matrix(powers, x_samples, evalF)

        # 3: 행렬식 연산
        if A_matrix.shape[0] != A_matrix.shape[1]:
            print(f"A가 직사각 행렬입니다. shape={A_matrix.shape[0]}, {A_matrix.shape[1]}")
            return err_return
        
        try:
            coeff, E = solve_matrix(A_matrix, y_matrix, n, powers)
        except np.linalg.LinAlgError:
            # print("Singular Matrix!!")
            return err_return

        # 4: 지역 극값 연산
        max_point_x, max_point_y = calculate_local_max(coeff, evalF, intervals)
        if -1 in max_point_y or 1 in max_point_y:
            return err_return
        
        if len(max_point_x) == sample_number:
            best_points_x = max_point_x
            best_points_y = max_point_y
        
        # 극점 개수가 너무 많은 경우
        elif len(max_point_x) > sample_number:
            extra_points = len(max_point_x) - sample_number
            debug_print(f"{extra_points} more local max points exists.", print_mode)
            # best_points_x, best_points_y = find_best_combination(max_point_x, max_point_y, sample_number, coeff, evalF, print_mode)
            best_points_x, best_points_y = select_max_points(max_point_x, max_point_y, sample_number, coeff, evalF, print_mode, E)
            if best_points_x == []:
                # best_points_x, best_points_y = select_max_points(max_point_x, max_point_y, sample_number, coeff, evalF, print_mode, E)
                return err_return

        # 극점 개수가 부족한 경우 - Deprecated.
        elif len(max_point_x) < sample_number:
            # print("MORE POINT NEEDED")
            return err_return
            # req_points = sample_number - len(max_point_x)
            # # print(f"Error\t\t\t: {req_points} more local max points required.")
            # debug_print(f"Error\t\t\t: {req_points} more local max points required.", print_mode)
            # # 각 구간의 양 끝점 중 오차가 큰 순서대로 삽입.
            # boundary_data = []
            # for start, end in intervals:
            #     boundary_data.append((start, error_abs(coeff, start, evalF(start))))
            #     boundary_data.append((end, error_abs(coeff, end, evalF(end))))
            # boundary_data.sort(key= lambda x: x[1])

            # added = 0
            # for x_val, err in reversed(boundary_data):
            #     if added == req_points:
            #         break
            #     x_np = np.float64(x_val)
            #     if x_np not in max_point_x:
            #         max_point_x.append(x_np)
            #         max_point_y.append(err)
            #         added += 1

            # # 2개의 점으로도 부족할 경우 오류를 출력하고 더미데이터 반환.
            # if added < req_points:
            #     debug_print(f"Fatal Error\t\t: {req_points-added} more points need to be added.", print_mode)
            #     best_points_x = max_point_x
            #     best_points_y = max_point_y
            #     return err_return
            # else:
            #     max_point_x.sort()
            #     max_point_y.sort()
            #     best_points_x = max_point_x
            #     best_points_y = max_point_y

        best_error_abs = [abs(x) for x in best_points_y]

        # 5: 종료조건 판단
        if decide_exit(best_error_abs, 1e-2, print_mode):
            try:
                if clni:
                    max_err, next_intervals = calculate_next_remez(coeff, evalF, eb, intervals)
                    if next_intervals[0][1] < next_intervals[1][0]:
                        return coeff, max_err, next_intervals
                    else:
                        debug_print("구간 대소관계 오류", print_mode)
                        return err_return
                else:
                    return coeff
            
            except Exception as e:
                print("구간 계산 오류")
                print(intervals)
                print(e)
                return err_return

        else:
            x_samples = best_points_x

        debug_print("", print_mode)
    
    debug_print("Approx failed.", print_mode)
    return err_return
