# векторизованные функции расчета pvt свойств 
from neftpy.uconvert import *
from neftpy.uconst import *

import neftpy.upvt_oil as pvto
import neftpy.upvt_gas as pvtg

import numpy as np


"""
для удобства работы с прикладными скриптами и ноутбуками
приводятся векторные реализации всех pvt функций

часть реализована в векторном виде, часть векторизуется принудительно
"""

""" 
====================================================================================================
Расчет давления насыщения
====================================================================================================
"""
# уже векторная реализация
unf_pb_Standing_MPaa = pvto.unf_pb_Standing_MPaa

unf_pb_Valko_MPaa = pvto.unf_pb_Valko_MPaa

unf_pb_Glaso_MPaa = pvto.unf_pb_Glaso_MPaa


""" 
====================================================================================================
Расчет газосодержания (Gas Solution Ratio) в нефти
====================================================================================================
"""

unf_rs_Standing_m3m3 = pvto.unf_rs_Standing_m3m3    # газосодержание по Стендингу

# наивная векторизация для старой версии
unf_rs_Velarde_2_m3m3 = np.vectorize(pvto._unf_rs_Velarde_m3m3_)

unf_rs_Velarde_m3m3 = pvto.unf_rs_Velarde_m3m3

unf_rsb_Mccain_m3m3 = pvto.unf_rsb_Mccain_m3m3


""" 
====================================================================================================
Расчет объемного коэффициента нефти (Oil Formation Volume Factor FVF)
====================================================================================================
"""

unf_bo_above_pb_m3m3 = pvto.unf_bo_above_pb_m3m3

unf_bo_below_pb_m3m3 = pvto.unf_bo_below_pb_m3m3

unf_bo_saturated_Standing_m3m3 = pvto.unf_bo_saturated_Standing_m3m3

unf_bo_saturated_Glaso_m3m3 = pvto.unf_bo_saturated_Glaso_m3m3

unf_bo_below_Glaso_m3m3 = pvto.unf_bo_below_Glaso_m3m3

""" 
====================================================================================================
Расчет плотности нефти
====================================================================================================
"""

unf_density_oil_Mccain = np.vectorize(pvto.unf_density_oil_Mccain)

unf_density_oil_Standing = pvto.unf_density_oil_Standing

""" 
====================================================================================================
Расчет сжимаемости нефти
====================================================================================================
"""

unf_compressibility_saturated_oil_McCain_1Mpa = pvto.unf_compressibility_saturated_oil_McCain_1Mpa

unf_compressibility_oil_VB_1Mpa = pvto.unf_compressibility_oil_VB_1Mpa

""" 
====================================================================================================
Расчет плотности газа на поверхности
====================================================================================================
"""

unf_gamma_gas_Mccain = pvto.unf_gamma_gas_Mccain

unf_deadoilviscosity_Beggs_cP = pvto.unf_viscosity_deadoil_Beggs_cP

""" 
====================================================================================================
Расчет вязкости нефти
====================================================================================================
"""

unf_viscosity_deadoil_Beggs_cP = pvto.unf_viscosity_deadoil_Beggs_cP

unf_viscosity_deadoil_BeggsRobinson_cP = pvto.unf_viscosity_deadoil_BeggsRobinson_cP

unf_viscosity_deadoil_Standing = pvto.unf_viscosity_deadoil_Standing

unf_viscosity_oil_Standing_cP = pvto.unf_viscosity_oil_Standing_cP

unf_viscosity_saturatedoil_Beggs_cP = pvto.unf_viscosity_saturatedoil_Beggs_cP


unf_viscosity_undersaturatedoil_VB_cP = pvto.unf_viscosity_undersaturatedoil_VB_cP
unf_viscosity_oil_Beggs_VB_cP = pvto.unf_viscosity_oil_Beggs_VB_cP
unf_viscosity_undersaturatedoil_Petrosky_cP = pvto.unf_viscosity_undersaturatedoil_Petrosky_cP

""" 
====================================================================================================
Расчет тепловых свойств
====================================================================================================
"""

unf_heat_capacity_oil_Gambill_JkgC = pvto.unf_heat_capacity_oil_Gambill_JkgC
unf_heat_capacity_oil_Wes_Wright_JkgC = pvto.unf_heat_capacity_oil_Wes_Wright_JkgC
unf_thermal_conductivity_oil_Abdul_Seoud_Moharam_WmK = pvto.unf_thermal_conductivity_oil_Abdul_Seoud_Moharam_WmK
unf_thermal_conductivity_oil_Smith_WmK = pvto.unf_thermal_conductivity_oil_Smith_WmK
unf_thermal_conductivity_oil_Cragoe_WmK = pvto.unf_thermal_conductivity_oil_Cragoe_WmK

unf_viscosity_saturatedoil_Beggs_cP = pvto.unf_viscosity_saturatedoil_Beggs_cP

""" 
====================================================================================================
Расчет свойств газа
====================================================================================================
"""


unf_pseudocritical_temperature_K = pvtg.unf_pseudocritical_temperature_K


unf_pseudocritical_pressure_MPa = pvtg.unf_pseudocritical_pressure_MPa

unf_pseudocritical_temperature_Standing_K = pvtg.unf_pseudocritical_temperature_Standing_K
unf_pseudocritical_pressure_Standing_MPa = pvtg.unf_pseudocritical_pressure_Standing_MPa

unf_zfactor_BrillBeggs = np.vectorize(pvtg.unf_zfactor_BrillBeggs)

unf_zfactor_DAK = np.vectorize(pvtg.unf_zfactor_DAK)
unf_zfactor_DAK_ppr = np.vectorize(pvtg.unf_zfactor_DAK_ppr)
unf_zfactor_Kareem = np.vectorize(pvtg.unf_zfactor_Kareem)
unf_zfactor_SK = np.vectorize(pvtg.unf_zfactor_SK)


unf_compressibility_gas_Mattar_1MPa = np.vectorize(pvtg.unf_compressibility_gas_Mattar_1MPa)

unf_gasviscosity_Lee_cP = pvtg.unf_gasviscosity_Lee_cP

unf_gas_fvf_m3m3 = pvtg.unf_gas_fvf_m3m3

unf_gas_density_kgm3 = pvtg.unf_gas_density_kgm3


unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC = pvtg.unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC


unf_thermal_conductivity_gas_methane_WmK = pvtg.unf_thermal_conductivity_gas_methane_WmK