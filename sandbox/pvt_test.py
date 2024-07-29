# тестируем функции для расчета PVT 

import neftpy.upvt_oil as pvto
import neftpy.upvt_np_vect as pvtovect

import numpy as np


rsb_m3m3 = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
gamma_oil = 0.86
gamma_gas = 0.6
t_K = 350
res = np.array([ 0.44434683,  1.67824568,  2.98339278, 20.1702107 , 35.85628831])

print(pvtovect.unf_pb_Valko_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K))
print(res)

print(np.allclose(pvtovect.unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), 
                        res, rtol=1e-05))