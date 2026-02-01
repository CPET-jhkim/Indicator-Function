from approx_remez_v2 import multi_remez_main
from polyEval.error_bound import ErrBound
from parse import build_desmos_script
from contextlib import redirect_stdout

# nohup /home/doodle/.virtualenvs/remez/bin/python /home/doodle/python/py_approx_test/main.py > /dev/null 2>&1 &
# pkill -u doodle -x python; pkill -u doodle -x python3

eb = ErrBound(sigma=3.1, N=pow(2,17), h=192, s=50)
e_num=30
filename = "output.txt"

with open(filename, 'w', encoding='utf-8') as f:
    with redirect_stdout(f):
        for p_num in range(2, 10):
            flag = multi_remez_main(p_num=p_num, e_num=e_num, max_n=9, eb=eb, print_mode="normal")
            if flag:
                print(build_desmos_script(f"coeff_{p_num}_{e_num}.txt"))
