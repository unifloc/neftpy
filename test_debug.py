#from pyrsistent import T
import neftpy.upvt as pvt
import neftpy.upvt_old as pvt_old
import numpy as np

""" 
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
"""

import neftpy.fluid as fl 

f = fl.BlackOilStanding(gamma_gas=0.6, gamma_oil=0.82)
f.set_calibration_param(pb_calibr_bar=100, tb_calibr_C=80, b_oilb_calibr_m3m3=1.1)

p = np.linspace(1, 100, 10)
f.calc(p_bar=p, t_C= 90)

print(f.pb_bar)
print(f.rs_m3m3)
print(f.b_oil_m3m3)