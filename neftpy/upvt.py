"""
PVT (pressure volume temperature) functions based on Black Oil PVT model for petroleum engineering calculations

Rinat Khabibullin
revision from 19/10/2021

Unifloc_VBA and unifloc_py refactoring
"""

import numpy as np
import neftpy.uconvert as uc

"""
Корреляции расчета давления насыщения
"""

# простая реализация (без векторизации)
def __unf_pb_Standing_MPaa__(rsb_m3m3, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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
    rsb_old = rsb_m3m3
    if rsb_m3m3 < min_rsb:
        rsb_m3m3 = min_rsb
    # мольная доля газа
    yg = 1.225 + 0.001648 * t_K - 1.769 / gamma_oil
    pb_MPaa = 0.5197 * (rsb_m3m3 / gamma_gas) ** 0.83 * 10 ** yg
    # для низких значений газосодержания зададим асимптотику Pb = 1 атма при Rsb = 0
    # для больших значений газосодержания не корректируем то, что дает корреляция
    if rsb_old < min_rsb:
        pb_MPaa = (pb_MPaa - 0.1013) * rsb_old / min_rsb + 0.1013  # 0.101325
    return pb_MPaa

# принудительно векторизованная версия 
_unf_pb_Standing_MPaa_ = np.vectorize(__unf_pb_Standing_MPaa__)

# векторизованная версия расчета 
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
                                      pb_MPaa
                    )

# не векторизованная версия
def __unf_pb_Valko_MPaa__(rsb_m3m3, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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
    rsb_old = rsb_m3m3
    if rsb_m3m3 < min_rsb:
        rsb_m3m3 = min_rsb
    if rsb_m3m3 > max_rsb:
        rsb_m3m3 = max_rsb

    z1 = -4.81413889469569 + 0.748104504934282 * np.log(rsb_m3m3) \
        + 0.174372295950536 * np.log(rsb_m3m3) ** 2 - 0.0206 * np.log(rsb_m3m3) ** 3
    api = uc.gamma_oil_2_api(gamma_oil)
    z2 = 1.27 - 0.0449 * api + 4.36 * 10 ** (-4) * api ** 2 - 4.76 * 10 ** (-6) * api ** 3
    z3 = 4.51 - 10.84 * gamma_gas + 8.39 * gamma_gas ** 2 - 2.34 * gamma_gas ** 3
    z4 = -7.2254661 + 0.043155 * t_K - 8.5548 * 10 ** (-5) * t_K ** 2 + 6.00696 * 10 ** (-8) * t_K ** 3
    z = z1 + z2 + z3 + z4
    lnpb = 2.498006 + 0.713 * z + 0.0075 * z ** 2
    pb_MPaa = 2.718282 ** lnpb
    """
    для низких значений газосодержания зададим асимптотику Pb = 1 атма при Rsb = 0
    корреляция Valko получена с использованием непараметрической регресии GRACE метод (SPE 35412)
    особенность подхода - за пределеми интервала адаптации ассимптотики не соблюдаются
    поэтому их устанавливаем вручную
    для больших значений газосодержания продолжим линейный тренд корреляции
    """
    if rsb_old < min_rsb:
        pb_MPaa = (pb_MPaa - 0.1013) * rsb_old / min_rsb + 0.1013
    if rsb_old > max_rsb:
        pb_MPaa = (pb_MPaa - 0.1013) * rsb_old / max_rsb + 0.1013

    return pb_MPaa

# наивная векторизация
_unf_pb_Valko_MPaa_ = np.vectorize(__unf_pb_Valko_MPaa__)

# векторизованная версия
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

    api = uc.gamma_oil_2_api(gamma_oil)
    z2 = 1.27 - 0.0449 * api + 4.36 * 10 ** (-4) * api ** 2 - 4.76 * 10 ** (-6) * api ** 3
    z3 = 4.51 - 10.84 * gamma_gas + 8.39 * gamma_gas ** 2 - 2.34 * gamma_gas ** 3
    z4 = -7.2254661 + 0.043155 * t_K - 8.5548 * 10 ** (-5) * t_K ** 2 + 6.00696 * 10 ** (
        -8) * t_K ** 3
    z = z1 + z2 + z3 + z4
    lnpb = 2.498006 + 0.713 * z + 0.0075 * z ** 2
    pb_MPaa = 2.718282 ** lnpb

    """
    for low values of gas content we set the asymptotics Pb = 1 atm with Rsb = 0
    the Valko correlation is obtained using the GRACE nonparametric regression method (SPE 35412)
    The peculiarity of this approach is that beyond the adaptation interval the asymptotics are not observed
    therefore they are set manually
    for large values of gas content we continue the linear trend of correlation
    """

    pb_MPaa = np.where(rsb_old < min_rsb, 
                                      (pb_MPaa - 0.1013) * rsb_old / min_rsb + 0.1013, 
                                      pb_MPaa)
    pb_MPaa = np.where(rsb_old > max_rsb, 
                                      (pb_MPaa - 0.1013) * rsb_old / max_rsb + 0.1013, 
                                      pb_MPaa)

    return pb_MPaa

    """
    Газосодержание
    Gas Sulotion Ratio 
    """

# простой расчет без векторизации
def __unf_rs_Standing_m3m3__(p_MPaa, pb_MPaa=0, rsb_m3m3=0, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
    """
        Gas-oil ratio calculation inverse of Standing (1947) correlation for bubble point pressure

    :param p_MPaa: pressure, MPa
    :param pb_MPaa: buble point pressure, MPa
    :param rsb_m3m3: gas-oil ratio at the bubble point pressure, m3/m3
    :param gamma_oil: specific oil density (by water)
    :param gamma_gas: specific gas density (by air)
    :param t_K: temperature, K
    :return: gas-oil ratio in m3/m3

    ref1 "A Pressure-Volume-Temperature Correlation for Mixtures of California Oil and Gases",
    M.B. Standing, Drill. & Prod. Prac., API, 1947.

    ref2  "Стандарт компании Юкос. Физические свойства нефти. Методы расчета." Афанасьев В.Ю., Хасанов М.М. и др. 2002 г
    может считать в случае если нет давления насыщения и газосодержания при давлении насыщения, корреляция не точная
    """

    if pb_MPaa == 0 or rsb_m3m3 == 0:
        # мольная доля газа
        yg = 1.225 + 0.001648 * t_K - 1.769 / gamma_oil
        rs_m3m3 = gamma_gas * (1.92 * p_MPaa / 10 ** yg) ** 1.204
    elif p_MPaa < pb_MPaa:
        rs_m3m3 = rsb_m3m3 * (p_MPaa / pb_MPaa) ** 1.204
    else:
        rs_m3m3 = rsb_m3m3
    return rs_m3m3

# наивная векторизация
_unf_rs_Standing_m3m3_ = np.vectorize(__unf_rs_Standing_m3m3__)

#todo надо сделать нормальную векторизацию _unf_rs_Standing_m3m3_ (простой тест сделан)
unf_rs_Standing_m3m3 = _unf_rs_Standing_m3m3_


def __unf_rs_Velarde_m3m3__(p_MPaa, pb_MPaa=10, rsb_m3m3=100., gamma_oil=0.86, gamma_gas=0.6, t_K=350):
    """
        Solution Gas-oil ratio calculation according to Velarde McCain (1999) correlation

    :param p_MPaa: pressure, MPa
    :param pb_MPaa: buble point pressure, MPa
    :param rsb_m3m3: gas-oil ratio at the bubble point pressure, m3/m3
    :param gamma_oil: specific oil density (by water)
    :param gamma_gas: specific gas density (by air)
    :param t_K: temperature, K
    :return: gas-oil ratio in m3/m3

        ref1 "Correlation of Black Oil Properties at Pressures Below Bubblepoint Pressure—A New Approach",
    J. VELARDE, T.A. BLASINGAME Texas A&M University, W.D. MCCAIN, JR. S.A. Holditch & Associates, Inc 1999

    """

    api = uc.gamma_oil_2_api(gamma_oil)
    t_F = uc.K_2_F(t_K)
    pb_psig = uc.MPa_2_psi(pb_MPaa) - 14.7
    if pb_psig > 0:
        pr = uc.MPa_2_psig(p_MPaa)  / pb_psig
    else:
        pr = 0

    if pr <= 0:
        rs_m3m3 = 0.0
    elif pr < 1:
        A = np.array([9.73 * 10 ** (-7), 1.672608, 0.929870, 0.247235, 1.056052])
        B = np.array([0.022339, -1.004750, 0.337711, 0.132795, 0.302065])
        C = np.array([0.725167, -1.485480, -0.164741, -0.091330, 0.047094])
        A0 = 9.73 * 10 ** (-7)
        A1 = 1.672608
        A2 = 0.929870
        A3 = 0.247235
        A4 = 1.056052
        B0 = 0.022339
        B1 = -1.004750
        B2 = 0.337711
        B3 = 0.132795
        B4 = 0.302065
        C0 = 0.725167
        C1 = -1.485480
        C2 = -0.164741
        C3 = -0.091330
        C4 = 0.047094
        a1 = A[0] * gamma_gas ** A[1] * api ** A[2] * t_F ** A[3] * pb_psig ** A[4]
        a2 = B[0] * gamma_gas ** B[1] * api ** B[2] * t_F ** B[3] * pb_psig ** B[4]
        a3 = C[0] * gamma_gas ** C[1] * api ** C[2] * t_F ** C[3] * pb_psig ** C[4]
        pr = uc.Pa_2_psig(p_MPaa * 10 ** 6)  / pb_psig
        rsr = a1 * pr ** a2 + (1 - a1) * pr ** a3
        rs_m3m3 = rsr * rsb_m3m3
    else:
        rs_m3m3 = rsb_m3m3
    return rs_m3m3

# наивная векторизация
_unf_rs_Velarde_m3m3_ = np.vectorize(__unf_rs_Velarde_m3m3__)

# заготовка под полную векторизацию
#todo надо сделать нормальную векторизацию _unf_rs_Standing_m3m3_ (простой тест сделан)
unf_rs_Velarde_m3m3 = _unf_rs_Velarde_m3m3_




def __unf_rsb_Mccain_m3m3__(rsp_m3m3, gamma_oil=0.86, psp_MPaa=0.0, tsp_K=0.0):
    """
        Solution Gas-oil ratio at bubble point pressure calculation according to McCain (2002) correlation
    taking into account the gas losses at separator and stock tank

    :param rsp_m3m3: separator producing gas-oil ratio, m3m3
    :param gamma_oil: specific oil density(by water)
    :param psp_MPaa: pressure in separator, MPaa
    :param tsp_K: temperature in separator, K
    :return: solution gas-oil ratio at bubble point pressure, rsb in m3/m3

    часто условия в сепараторе неизвестны, может считать и без них по приблизительной формуле

    ref1 "Reservoir oil bubblepoint pressures revisited; solution gas–oil ratios and surface gas specific gravities",
    J. VELARDE, W.D. MCCAIN, 2002
    """

    rsp_scfstb = uc.m3m3_2_scfstb(rsp_m3m3)
    if psp_MPaa > 0 and tsp_K > 0:
        api = uc.gamma_oil_2_api(gamma_oil)
        psp_psia = uc.Pa_2_psi(psp_MPaa * 10 ** 6)
        tsp_F = uc.K_2_F(tsp_K)
        z1 = -8.005 + 2.7 * np.log(psp_psia) - 0.161 * np.log(psp_psia) ** 2
        z2 = 1.224 - 0.5 * np.log(tsp_F)
        z3 = -1.587 + 0.0441 * np.log(api) - 2.29 * 10 ** (-5) * np.log(api) ** 2
        z = z1 + z2 + z3
        rst_scfstb = np.exp(3.955 + 0.83 * z - 0.024 * z ** 2 + 0.075 * z ** 3)
        rsb = rsp_scfstb + rst_scfstb
    elif rsp_m3m3 >= 0:
        rsb = 1.1618 * rsp_scfstb
    else:
        rsb = 0
    rsb = uc.scfstb_2_m3m3(rsb)
    return rsb

_unf_rsb_Mccain_m3m3_ = np.vectorize(__unf_rsb_Mccain_m3m3__)

unf_rsb_Mccain_m3m3 = np.vectorize(__unf_rsb_Mccain_m3m3__)