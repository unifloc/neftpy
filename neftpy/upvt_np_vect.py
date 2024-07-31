# векторизованные функции расчета pvt свойств 
from neftpy.uconvert import *
from neftpy.uconst import *

import neftpy.upvt_oil as pvto

import numpy as np


"""
для удобства работы с прикладными скриптами и ноутбуками
приводятся векторные реализации всех pvt функций

часть реализована в векторном виде, часть векторизуется принудительно
"""

# уже векторная реализация
unf_pb_Standing_MPaa = pvto.unf_pb_Standing_MPaa

unf_pb_Valko_MPaa = pvto.unf_pb_Valko_MPaa

unf_rs_Standing_m3m3 = pvto.unf_rs_Standing_m3m3    # газосодержание по Стендингу

# наивная векторизация для старой версии
unf_rs_Velarde_2_m3m3 = np.vectorize(pvto._unf_rs_Velarde_m3m3_)

unf_rs_Velarde_m3m3 = pvto.unf_rs_Velarde_m3m3

unf_rsb_Mccain_m3m3 = pvto.unf_rsb_Mccain_m3m3

unf_bo_above_pb_m3m3 = pvto.unf_bo_above_pb_m3m3

unf_bo_below_pb_m3m3 = pvto.unf_bo_below_pb_m3m3

unf_bo_saturated_Standing_m3m3 = pvto.unf_bo_saturated_Standing_m3m3

unf_density_oil_Mccain = pvto.unf_density_oil_Mccain

unf_density_oil_Standing = pvto.unf_density_oil_Standing

unf_compressibility_saturated_oil_McCain_1Mpa = pvto.unf_compressibility_saturated_oil_McCain_1Mpa

unf_compressibility_oil_VB_1Mpa = pvto.unf_compressibility_oil_VB_1Mpa

unf_gamma_gas_Mccain = pvto.unf_gamma_gas_Mccain

unf_deadoilviscosity_Beggs_cP = pvto.unf_deadoilviscosity_Beggs_cP

unf_saturatedoilviscosity_Beggs_cP = pvto.unf_saturatedoilviscosity_Beggs_cP

unf_undersaturatedoilviscosity_VB_cP = pvto.unf_undersaturatedoilviscosity_VB_cP

unf_oil_viscosity_Beggs_VB_cP = pvto.unf_oil_viscosity_Beggs_VB_cP

unf_heat_capacity_oil_Gambill_JkgC = pvto.unf_heat_capacity_oil_Gambill_JkgC

unf_thermal_conductivity_oil_Cragoe_WmK = pvto.unf_thermal_conductivity_oil_Cragoe_WmK