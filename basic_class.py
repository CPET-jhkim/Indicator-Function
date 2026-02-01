# basic_class.py
from polyEval.algorithm import cal_polyEval

class CalData:
    def __init__(self, title="", iter=0):
        self.title = title
        self.iter = iter
        self.depth = 0
        self.add_count = 0
        self.pmult = 0
        self.cmult = 0

    def add(self, c2: 'CalData', mode="add"):
        if mode == "add":
            self.depth += c2.depth
        elif mode == "compare":
            self.depth = self.depth if self.depth >= c2.depth else c2.depth
        self.add_count += c2.add_count
        self.pmult += c2.pmult
        self.cmult += c2.cmult
        
    def __lt__(self, other):
        return (self.depth, self.cmult, self.pmult, self.add_count) < (other.depth, other.cmult, other.pmult, other.add_count)
    
    def __eq__(self, other):
        return (self.depth, self.cmult, self.pmult, self.add_count) == (other.depth, other.cmult, other.pmult, other.add_count)
    def __le__(self, other):
        return self < other or self == other
    
    def print_params(self, title=False, iter=False):
        if title:
            print(f"Title:\t\t{self.title}")
        if iter:
            print(f"Iter:\t\t{self.iter}")
        print(f"DCPA: {self.depth}|{self.cmult}|{self.pmult}|{self.add_count}")


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

    def print_params(self):
        print(f"precision: ", end='')
        print(' -> '.join(map(str, self.pre_log)))
        print(f"coeff: ", end='')
        print(' -> '.join(map(str, [len(s)-1 for s in self.coeff_log])))
        # print(f"total complexity:")
        self.total_CalData.print_params()
        print("#"*20)
        
# 함수 종류 확인
def detect_coeff_type(coeff: list[float]) -> str:
    has_even = any(abs(c) != 0.0 for i, c in enumerate(coeff) if i % 2 == 0)
    has_odd  = any(abs(c) != 0.0 for i, c in enumerate(coeff) if i % 2 == 1)

    if has_even and not has_odd:
        return "even"
    if has_odd and not has_even:
        return "odd"
    return "all"

# 특정 함수의 반복횟수에 대한 연산량, 깊이 계산
def cal_iter(coeff: list[float], iter: int) -> CalData:
    res = cal_coeff(coeff)
    res.depth *= iter
    res.add_count *= iter
    res.cmult *= iter
    res.pmult *= iter

    return res


# 특정 함수의 연산량, 깊이 계산
def cal_coeff(coeff: list[float]) -> CalData:
    cmp = cal_polyEval(coeff)
    assert type(cmp) is not bool
    cmp = cmp.comp
    cal = CalData()
    cal.depth = cmp.depth
    cal.cmult = cmp.cmult
    cal.pmult = cmp.pmult
    cal.add_count = cmp.add
    return cal
    # # 함수가 홀수/짝수/전체인지 확인
    # coeff_type = detect_coeff_type(coeff)
    # res = CalData()
    # max_deg = len(coeff) - 1

    # if coeff_type == "all":
    #     res.cmult = max_deg - 1
    #     res.add_count = max_deg
    # elif coeff_type == "even":
    #     res.cmult = int(max_deg / 2)
    #     res.add_count = int(max_deg / 2)

    # res.pmult = 1
    # res.depth = res.cmult + 1
    # return res
