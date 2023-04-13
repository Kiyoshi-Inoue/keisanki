import random
import numpy as np

def gene_u_i(p_i,r_J):
    vec_u_i= []
    for i in range(p_i*r_J):
        x = random.uniform(0.5, 1)  # 0.5から1までの範囲で一様分布に従う乱数を生成する
        vec_u_i.append(x)
    U_i=np.array(vec_u_i).reshape(p_i,r_J)
    return U_i
