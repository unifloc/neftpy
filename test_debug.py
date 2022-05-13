import neftpy.upvt as pvt
import neftpy.upvt_old as pvt_old
import numpy as np

p_MPaa =  np.array([0.1, 1,5,10, 20, 30])
pb_MPaa = 10
co_1MPa = 3 * 10**(-3)
rs_m3m3 = 150
gamma_gas = 0.6
t_K = 350
gamma_oil = 0.86
gamma_gassp = 0
res = pvt.unf_density_oil_Mccain(p_MPaa, pb_MPaa, co_1MPa, rs_m3m3, gamma_gas, t_K, gamma_oil,gamma_gassp)

print(res)

res = pvt_old.unf_density_oil_Mccain(p_MPaa, pb_MPaa, co_1MPa, rs_m3m3, gamma_gas, t_K, gamma_oil,gamma_gassp)

print(res)