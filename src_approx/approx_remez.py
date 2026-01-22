import numpy as np
from src_remez.algorithm import remez_algorithm, cleanse
from src_remez.print_plot import *
from src_remez.print import print_intervals, debug_print, coeff2txt, interval2txt
from src_remez.basic import calculate_next_remez
from src_errbound.error_bound import EB
from cal_depth import cal_coeff, CalData
from math import log2
import copy
from polyEval.algorithm import cal_polyEval

class remezData:
    def __init__(self, interval):
        self.coeff_log: list = []
        self.interval_log: list = [interval]
        self.CalData_log: list[CalData] = []
        self.pre_log: list[int] = []
        self.total_CalData = CalData()
        self.iter_cleanse = 0

    def update(self, coeff, pre, interval):
        self.add(coeff)
        self.pre_log.append(pre)
        self.interval_log.append(interval)

    def add(self, coeff: list):
        self.coeff_log.append(coeff)
        cal = cal_coeff(coeff)
        self.CalData_log.append(cal)
        self.total_CalData.add(cal, "add")

    def __lt__(self, other) -> bool:
        return self.total_CalData < other.total_CalData and len(self.pre_log) < len(other.pre_log)
    def __eq__(self, other) -> bool:
        return self.total_CalData == other.total_CalData and len(self.pre_log) == len(other.pre_log)
    def __gt__(self, other) -> bool:
        return self.total_CalData > other.total_CalData and len(self.pre_log) > len(other.pre_log)
    
    # def compare(self, challenger, eb: EB) -> int:
    #     if type(challenger) == list:
    #         challenger_cal = cal_coeff(challenger)
    #         res = self.total_CalData.compare(challenger_cal, "both")
    #         return res
    
    #     elif type(challenger) == remezData:
    #         res = self.total_CalData.compare(challenger.total_CalData, "both")
    #         if res == 0:
    #             if len(self.coeff_log) < len(challenger.coeff_log):
    #                 return 1
    #             elif len(self.coeff_log) > len(challenger.coeff_log):
    #                 return -1
    #             else:
    #                 if self.pre_log[-1] < challenger.pre_log[-1]:
    #                     return 1
    #                 elif self.pre_log[-1] > challenger.pre_log[-1]:
    #                     return -1
    #         return res

    """
        새로운 비교방안????
        1. 동일하게 각각 계산복잡도 비교
        2. 동일한 depth ->  
    """
    # def compare(self, challenger: 'remezData') -> int:
    #     pass

    def print_params(self):
        print(f"precision: ", end='')
        for pre in self.pre_log:
            print(f"{pre}", end='')
            if pre != self.pre_log[-1]:
                print(" -> ", end='')
            else:
                print()
        print(f"coeff: ", end='')
        for coeff in self.coeff_log:
            print(f"{len(coeff)-1}", end='')
            if coeff != self.coeff_log[-1]:
                print(" -> ", end='')
            else:
                print()
        print(f"total complexity:")
        self.total_CalData.print_params()
        print("#"*20)
    
def remez_recursion(routes: list, data: remezData, e_num: int, max_n: int, eb: EB, approx_mode: str, printmode="debug"):
    n = 2
    intervals = data.interval_log[-1]

    def evalF(x):
        return 1 if x >= 0.5 else 0

    while n <= max_n:
        coeff, max_err, next_intervals = remez_algorithm(n, intervals, evalF, approx_mode, eb, 2, "normal")

        pre = -int(log2(max_err))
        if coeff == [-1]:
            break
        if pre < 0:
            debug_print(f"precision < 0", printmode)
            break
        
        debug_print(f"n={n}, pre {pre}, step {len(data.coeff_log) + 1}, intervals: ", printmode, "")
        if printmode == "debug":
            print_intervals(next_intervals, 10)

        # 새로운 remezData 객체 생성 후 정보 갱신
        newData = copy.deepcopy(data)
        newData.update(coeff, pre, next_intervals)

        if pre >= e_num: # 성공
            routes.append(newData)
            return
        
        # 정밀도가 목표의 2/3이상 도달한 경우 Cleanse 고려
        if pre >= e_num * (2/3):
            newData_cl = newData
            for iter in range(3):
                max_err, next_intervals = cleanse(eb, next_intervals)
                pre = -int(log2(max_err))

                debug_print(f"n={n}, pre {pre}, step {len(data.coeff_log) + 1} +  cleanse {iter+1}\n \t\tintervals: ", printmode, "")
                if printmode == "debug":
                    print_intervals(next_intervals, 10)

                newData_cl = copy.deepcopy(newData_cl)
                newData_cl.update([0, 0, 3, -2], pre, next_intervals)
                newData_cl.iter_cleanse += 1
                if pre >= e_num:
                    routes.append(newData_cl)
                    return
                
        remez_recursion(routes, newData, e_num, max_n, eb, approx_mode, printmode) # 새 객체가 정확도 조건을 만족할 때까지 재귀

        if approx_mode == "all":
            n += 1
        elif approx_mode == "even":
            n += 2
    return

def remez_recursion_v2(routes: list, data: remezData, e_num: int, max_n: int, eb: EB, printmode="debug"):
    n = 2
    intervals = data.interval_log[-1]
    if intervals[0][1] >= 0.5 or intervals[1][0] < 0.5:
        return
    
    def evalF(x):
        return 1 if x >= 0.5 else 0

    while n <= max_n:
        coeff = remez_algorithm(n, intervals, evalF, "all", eb, clni=False)
        if coeff == [-1] or coeff[0] == [-1]:
            n += 1
            continue
            
        # 계수의 크기가 작은 값부터 제거하면서 연산.
        # 효율성이 확실히 낮은 데이터는 제거
        def remove_coeff(coeff, k) -> list:
            # 기존 0이 아닌 계수들만 대상으로 인덱스 수집
            nonzero_idxs = [i for i, c in enumerate(coeff) if c != 0.0]

            # 절댓값 기준으로 작은 순서대로 k개 선택
            idxs = sorted(nonzero_idxs, key=lambda i: abs(coeff[i]))[:k]

            # 제거(0으로 설정)
            tc = coeff.copy()
            for i in idxs:
                tc[i] = 0.0

            return tc
        
        def prune_dominated_by_sort(rem_result):
            # dcmp 오름차순, pre 내림차순
            items = sorted(
                rem_result.items(),
                key=lambda kv: (-kv[1][1], kv[1][0])
            )

            kept = {}
            max_pre_so_far = -float("inf")

            for k, (dcmp, pre, next_intervals) in items:
                # 이미 더 작은 dcmp에서 더 큰 pre를 본 적이 있으면 지배됨
                if pre < max_pre_so_far:
                    continue

                kept[k] = (dcmp, pre, next_intervals)
                max_pre_so_far = max(max_pre_so_far, pre)

            return kept
        
        rem_result = {}
        for k in range(n, -1, -1):
            temp_coeff = remove_coeff(coeff, k)
            max_err, next_intervals = calculate_next_remez(temp_coeff, evalF, eb, intervals)
            pre = -int(log2(max_err))
            if pre < 0:
                debug_print(f"precision < 0", printmode)
                continue
            if next_intervals[0][1] >= 0.5 or next_intervals[1][0] < 0.5:
                continue
            dcmp = cal_polyEval(temp_coeff)
            if type(dcmp) != bool:
                rem_result[k] = (dcmp, pre, next_intervals)
        # 선정
        res = prune_dominated_by_sort(rem_result)

        # debug
        # print(f"data count: {len(res.values())}")
        # for dcmp, pre, next_intervals in res.values():
        #     print(f"pre: {pre}, coeff: {dcmp.coeff} ", end='')
        #     dcmp.comp.print_params()
            
        for key, value in res.items():
            dcmp, pre, next_intervals = value
            try:       
                debug_print(f"n={n}, pre {pre}, step {len(data.coeff_log) + 1}, rm {key}, intervals: ", printmode, "")
                if printmode == "debug":
                    print_intervals(next_intervals, 10)

                # 새로운 remezData 객체 생성 후 정보 갱신
                newData = copy.deepcopy(data)
                newData.update(dcmp.coeff, pre, next_intervals)

                if pre >= e_num: # 성공
                    if len(routes) == 0 or newData.total_CalData <= routes[0].total_CalData:
                        routes.append(newData)
                        routes.sort()
                        routes[0].total_CalData.print_params()
                        for c in routes[0].coeff_log:
                            print(c)
                    break
                else:
                    # 지속가능성 검증
                    routes.sort()
                    if len(routes) == 0 or newData.total_CalData < routes[0].total_CalData:
                        remez_recursion_v2(routes, newData, e_num, max_n, eb, printmode)
                        
            except Exception as e:
                print(e)
                # print(f"dcmp: {dcmp}, pre: {pre}, ni: {next_intervals}")
        n += 1
    return

def multi_remez(p_num: int, e_num: int,max_n: int, eb: EB, approx_mode: str, print_mode: str, recursion_mode: int):    
    p = pow(2, p_num)
    e = eb.Bc/eb.scale    
    intervals = [[0, e], [(1-e), (p-1+e)]]   
    
    routes = [] # 최종 경로 목록

    # 초기 evalF
    def evalF(x):
        return 1 if x < 0.5 else 0
        
    # 첫구간 remez 수행 후 재귀
    for n in range(2, max_n+1, 2):
        coeff, max_err, next_intervals = remez_algorithm(n, intervals, evalF, approx_mode, eb, 2, "normal")
        pre = -int(log2(max_err))

        debug_print(f"Init approximation\nn={n}, step 0, intervals: ", print_mode, "")
        if print_mode == "debug":
            print_intervals(next_intervals, 10)

        # remezData 객체 생성 후 정보 갱신
        data = remezData(intervals)
        data.update(coeff, pre, next_intervals)

        # 재귀
        if recursion_mode == 1:
            remez_recursion(routes, data, e_num, max_n, eb, approx_mode, print_mode)
        elif recursion_mode == 2:
            remez_recursion_v2(routes, data, e_num, max_n, eb, "normal")
    
    if routes[0].pre_log[-1] >= e_num: # 성공
        print(f"Total {len(routes)} routes.")

        # 최종 비교
        routes.sort()
        best = routes[0]

        # 텍스트파일 저장
        # for i in range(len(bob)):
        with open(f"coeff_{p_num}_{e_num}.txt", "w", encoding="utf-8") as f:
            for coeff in best.coeff_log:
                f.write(coeff2txt(coeff))
                f.write('\n')
        
        print(f"(p, e): {p_num}, {e_num}")
        best.print_params()
    
    else: # 실패
        print("Approximation Failed.")
        routes[0].print_params()