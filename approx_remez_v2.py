# Remez without recursion
import numpy as np
from math import log2
import copy, traceback, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from src_remez.algorithm import remez_algorithm
from src_remez.print_plot import *
from src_remez.print import print_intervals, debug_print, coeff2txt
from src_remez.basic import calculate_next_remez
from basic_class import remezData, CalData
from polyEval.error_bound import ErrBound
from polyEval.algorithm import cal_polyEval

def evalF_step1(x):
    return 0 if x >= 0.5 else 1

def evalF_step2(x):
    return 1 if x >= 0.5 else 0

def multi_remez_main(p_num: int, e_num: int, max_n: int, eb: ErrBound, print_mode: str):
    p = pow(2, p_num)
    e = float(eb.Bc/eb.scale)
    intervals = [[0, e], [(1-e), (p-1+e)]]   
    
    all_completed_routes = []

    # Step1 - 각 n에 대하여 초기 근사를 진행
    ongoing_routes = []
    for n in range(2, max_n+1, 2):
        coeff, max_err, next_intervals = remez_algorithm(n, intervals, evalF_step1, "even", eb, "normal")
        if coeff == [-1]:
            print(f"Init approximation\nn={n} failed.")
            return 
        else:
            print('-' * 30)
            print(f"Init approximation\nn={n}, step 0, intervals: ")
            print_intervals(next_intervals, 10)
            
        pre = -(log2(max_err))
        data = remezData(intervals)
        data.update(coeff, pre, next_intervals)
        ongoing_routes.append(data)
    
    # Step2 - 반복
    step_count = 1
    while len(ongoing_routes) > 0:
        print(f"Step {step_count}: Processing {len(ongoing_routes)} routes...")
        new_completed, new_ongoing = multi_remez_sub(ongoing_routes, e_num, max_n, eb, print_mode)
        
        all_completed_routes.extend(new_completed)
        ongoing_routes = new_ongoing
        step_count += 1
        
    # Step3 - 최종 비교
    if len(all_completed_routes) > 0: # 성공
        print(f"Total {len(all_completed_routes)} routes found.")

        # 최종 비교 및 정렬
        all_completed_routes.sort()
        best = all_completed_routes[0]

        # 텍스트파일 저장
        filename = f"coeff_{p_num}_{e_num}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for coeff in best.coeff_log:
                f.write(coeff2txt(coeff))
                f.write('\n')
        
        print(f"Saved best solution to {filename}")
        print(f"(p, e): {p_num}, {e_num}")
        best.print_params()
        return True
    
    else: # 실패
        print("Approximation Failed.")
        return False

def multi_remez_sub(ongoing_routes: list, e_num: int, max_n: int, eb: ErrBound, print_mode: str):
    '''
    submodule
    routes의 각 데이터는 멀티코어(최대 32개)를 사용하여 수행.
    '''
    completed_routes = []
    new_routes = []
    
    num_tasks = len(ongoing_routes)
    max_workers = min(32, num_tasks) if num_tasks > 0 else 1
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 각 route 확장을 병렬 작업으로 제출
        futures = [
            executor.submit(remez_step, route, e_num, max_n, eb)
            for route in ongoing_routes
        ]
        
        for future in as_completed(futures):
            try:
                local_completed, local_new = future.result()
                completed_routes.extend(local_completed)
                new_routes.extend(local_new)
            except Exception as e:
                print(f"Worker failed with error: {e}")
                traceback.print_exc()

    return completed_routes, new_routes


# 중단여부 검사
def check_continue(base: remezData, route: remezData) -> bool:
    # 조건1: 정밀도
    if base.pre_log[-1] > route.pre_log[-1]:
        return False
    # 조건2: 과도한 반복
    if len(route.coeff_log) >= 10:
        return False
    # 조건3: Cleanse
    if route.coeff_log[-1] == [0, 0, 3, -2]:
        return False
    return True  

# 개별 프로세스
def remez_step(route: remezData, e_num: int, max_n: int, eb: ErrBound):
    local_completed = []
    local_new = []
    
    # 1. CL
    try:
        intervals = route.interval_log[-1]
        cl_coeff = [0, 0, 3, -2]
        max_err, next_intervals = calculate_next_remez(cl_coeff, evalF_step2, eb, intervals)
        pre = min(-(log2(max_err)), e_num)
        
        data = copy.deepcopy(route)
        data.update(cl_coeff, pre, next_intervals)
        
        if check_continue(route, data):
            if pre >= e_num:
                local_completed.append(data)
            else:
                local_new.append(data)
    except Exception:
        pass

    # 2. All, Even, Odd
    for n in range(2, max_n + 1):
        modes_to_try = ["all"]
        if n % 2 == 0:
            modes_to_try.append("even")
        else:
            modes_to_try.append("odd")

        for mode in modes_to_try:
            try:
                intervals = route.interval_log[-1]
                coeff, max_err, next_intervals = remez_algorithm(n, intervals, evalF_step2, mode, eb, "normal")

                if coeff == [-1]:
                    continue

                pre = min(-(log2(max_err)), e_num)
                
                data = copy.deepcopy(route)
                data.update(coeff, pre, next_intervals)
                
                if check_continue(route, data):
                    if pre >= e_num:
                        local_completed.append(data)
                    else:
                        local_new.append(data)
            except Exception:
                continue
                
    return local_completed, local_new