from src_approx.approx_remez import multi_remez
from src_errbound.error_bound import EB
from parse import build_desmos_script

p_num = 2
e_num = 30

p = pow(2, p_num)
e = pow(2, -e_num)

interval_nonScale =  [[-p, -(1-e)],     [-e, e],        [(1-e), p-1]]
interval_Scale =     [[-1, -(1-e)/p],   [-e/p, e/p],    [(1-e)/p, 1-1/p]]

criteria = "ctxt" # depth, ctxt

max_n = 6
eb = EB(sigma=3.1, N=pow(2,17), h=192, s=50)

multi_remez(p_num, e_num, max_n, eb, "even", "normal", 2)
print(build_desmos_script(f"coeff_{p_num}_{e_num}.txt"))

# for p_num in [5, 6, 7, 8, 9]:
#     for e_num in [30]:
#         try:
#             # sgn(p_num, e_num, interval_Scale, criteria)
#             multi_remez(p_num, e_num, max_n, eb, approx_mode, "normal")
#             print(build_desmos_script(f"coeff_{p_num}_{e_num}.txt"))
#         except Exception as e:
#             # print("main")   
#             print(e)
#             continue

# multi_remez(p_num, e_num, max_n, False, "normal")
# print(build_desmos_script(f"coeff_{p_num}_{e_num}.txt"))