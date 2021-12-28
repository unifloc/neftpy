"""
PVT (pressure volume temperature) functions based on Black Oil PVT model for petroleum engineering calculations

Rinat Khabibullin
revision from 19/10/2021

Unifloc_VBA and unifloc_py refactoring
"""

import numpy as np
import neftpy.uconvert as uc


def unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
    """
        bubble point pressure calculation according to Standing (1947) correlation

    :param rsb_m3m3: solution ration at bubble point, must be given, m3/m3
    :param gamma_oil: specific oil density (by water)
    :param gamma_gas: specific gas density (by air)
    :param t_K: temperature, K
    :return: bubble point pressure abs in MPa

    ref1 "A Pressure-Volume-Temperature Correlation for Mixtures of California Oil and Gases",
    M.B. Standing, Drill. & Prod. Prac., API, 1947.

    ref2  "Стандарт компании Юкос. Физические свойства нефти. Методы расчета." Афанасьев В.Ю., Хасанов М.М. и др. 2002 г
    """

    min_rsb = 1.8
    rsb_old = np.copy(rsb_m3m3)
    rsb_m3m3 = np.where(rsb_m3m3 < min_rsb, min_rsb, rsb_m3m3)
    yg = 1.225 + 0.001648 * t_K - 1.769 / gamma_oil         # gas molar fraction
    pb_MPaa = 0.5197 * (rsb_m3m3 / gamma_gas) ** 0.83 * 10 ** yg
    # for low rsb values, we set the asymptotics Pb = 1 atma at Rsb = 0
    # for large rsb values do not correct what the correlation gives
    return np.where(rsb_old < min_rsb, 
                                      (pb_MPaa - 0.1013) * rsb_old / min_rsb + 0.1013, 
                                      pb_MPaa)


def unf_pb_Valko_MPaa(rsb_m3m3, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
    """
        bubble point pressure calculation according to Valko McCain (2002) correlation

    :param rsb_m3m3: solution ration at bubble point, must be given, m3/m3
    :param gamma_oil: specific oil density (by water)
    :param gamma_gas: specific gas density (by air)
    :param t_K: temperature, K
    :return: bubble point pressure abs in MPa

    ref SPE  "Reservoir oil bubblepoint pressures revisited; solution gas–oil ratios and surface gas specific gravities"
    W. D. McCain Jr.,P.P. Valko,
    """

    min_rsb = 1.8
    max_rsb = 800
    rsb_old = np.copy(rsb_m3m3)
    rsb_m3m3 = np.where(rsb_m3m3 < min_rsb, min_rsb, rsb_m3m3)
    rsb_m3m3 = np.where(rsb_m3m3 > max_rsb, max_rsb, rsb_m3m3)
 
    z1 = -4.81413889469569 + 0.748104504934282 * np.log(rsb_m3m3) \
        + 0.174372295950536 * np.log(rsb_m3m3) ** 2 - 0.0206 * np.log(rsb_m3m3) ** 3

    #init
    #z2 = 25.537681965 - 57.519938195 / gamma_oil + 46.327882495 / gamma_oil ** 2 - 13.485786265 / gamma_oil ** 3 init
    #vba
    api = uc.gamma_oil2api(gamma_oil)
    z2 = 1.27 - 0.0449 * api + 4.36 * 10 ** (-4) * api ** 2 - 4.76 * 10 ** (-6) * api ** 3
    z3 = 4.51 - 10.84 * gamma_gas + 8.39 * gamma_gas ** 2 - 2.34 * gamma_gas ** 3

    #init
    #z4 = 6.00696e-8 * t_K ** 3 - 8.554832172e-5 * t_K ** 2 + 0.043155018225018 * t_K - 7.22546617091445
    # vba
    z4 = -7.2254661 + 0.043155 * t_K - 8.5548 * 10 ** (-5) * t_K ** 2 + 6.00696 * 10 ** (
        -8) * t_K ** 3

    z = z1 + z2 + z3 + z4

    #init
    #pb_atma = 119.992765886175 * np.exp(0.0075 * z ** 2 + 0.713 * z)
    #pb_MPaa = pb_atma / 10.1325
    #vba
    lnpb = 2.498006 + 0.713 * z + 0.0075 * z ** 2
    pb_MPaa = 2.718282 ** lnpb
    """
    для низких значений газосодержания зададим асимптотику Pb = 1 атма при Rsb = 0
    корреляция Valko получена с использованием непараметрической регресии GRACE метод (SPE 35412)
    особенность подхода - за пределеми интервала адаптации ассимптотики не соблюдаются
    поэтому их устанавливаем вручную
    для больших значений газосодержания продолжим линейный тренд корреляции
    """

    pb_MPaa = np.where(rsb_old < min_rsb, 
                                      (pb_MPaa - 0.1013) * rsb_old / min_rsb + 0.1013, 
                                      pb_MPaa)
    pb_MPaa = np.where(rsb_old > max_rsb, 
                                      (pb_MPaa - 0.1013) * rsb_old / max_rsb + 0.1013, 
                                      pb_MPaa)

    return pb_MPaa