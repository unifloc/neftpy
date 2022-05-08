"""
PVT (pressure volume temperature) functions based on Black Oil PVT model for petroleum engineering calculations

Rinat Khabibullin
revision from 19/10/2021

Unifloc_VBA and unifloc_py refactoring
"""

import numpy as np
import neftpy.uconvert as uc
import scipy.optimize as opt

"""
Корреляции расчета давления насыщения
"""

# простая реализация (без векторизации)
def __unf_pb_Standing_MPaa__(rsb_m3m3=100, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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
def unf_pb_Standing_MPaa(rsb_m3m3=100, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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
def __unf_pb_Valko_MPaa__(rsb_m3m3=100, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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
    z2 = 25.537681965 - 57.519938195 / gamma_oil + 46.327882495 / gamma_oil**2 \
         - 13.485786265 / gamma_oil ** 3
    z3 = 4.51 - 10.84 * gamma_gas + 8.39 * gamma_gas ** 2 - 2.34 * gamma_gas ** 3
    z4 = -7.2254661 + 0.043155 * t_K - 8.5548e-5 * t_K ** 2 + 6.00696e-8 * t_K ** 3
    z = z1 + z2 + z3 + z4

    pb_MPaa = 12.1582266504102 * np.exp(0.0075 * z**2 + 0.713 * z)

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
def unf_pb_Valko_MPaa(rsb_m3m3=100, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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
    nplogrsb_m3m3 = np.log(rsb_m3m3)
    z1 = -4.81413889469569 + 0.748104504934282 * nplogrsb_m3m3 \
        + 0.174372295950536 * nplogrsb_m3m3 ** 2 - 0.0206 * nplogrsb_m3m3 ** 3
    z2 = 25.537681965 - 57.519938195 / gamma_oil + 46.327882495 / gamma_oil**2 \
         - 13.485786265 / gamma_oil ** 3
    z3 = 4.51 - 10.84 * gamma_gas + 8.39 * gamma_gas ** 2 - 2.34 * gamma_gas ** 3
    z4 = -7.2254661 + 0.043155 * t_K - 8.5548e-5 * t_K ** 2 + 6.00696e-8 * t_K ** 3
    z = z1 + z2 + z3 + z4

    pb_MPaa = 12.1582266504102 * np.exp(0.0075 * z**2 + 0.713 * z)

    """
    for low values of gas content we set the asymptotics Pb = 1 atm with Rsb = 0
    the Valko correlation is obtained using the GRACE nonparametric regression method (SPE 35412)
    The peculiarity of this approach is that beyond the adaptation interval the asymptotics are not observed
    therefore they are set manually
    for large values of gas content we continue the linear trend of correlation
    """

    pb_MPaa = np.where(rsb_old < min_rsb, 
                       (pb_MPaa - 0.1013) * rsb_old / min_rsb + 0.1013, 
                       np.where(rsb_old > max_rsb, 
                                (pb_MPaa - 0.1013) * rsb_old / max_rsb + 0.1013, 
                                pb_MPaa)
                        )

    return pb_MPaa

    """
    Газосодержание
    Gas Sulotion Ratio 
    """

# простой расчет без векторизации
def __unf_rs_Standing_m3m3__(p_MPaa=1, pb_MPaa=10, rsb_m3m3=0, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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

# расчет c векторизацией
def unf_rs_Standing_m3m3(p_MPaa=1, pb_MPaa=10, rsb_m3m3=100, gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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
    
    yg = lambda t_K, gamma_oil: 1.225 + 0.001648 * t_K - 1.769 / gamma_oil
    rs_m3m3 = np.where(pb_MPaa * rsb_m3m3 == 0, 
                        gamma_gas * (1.92 * p_MPaa / 10 ** yg(t_K, gamma_oil)) ** 1.204, 
                        np.where(p_MPaa < pb_MPaa, 
                                rsb_m3m3 * np.divide(p_MPaa, pb_MPaa, 
                                                     where=pb_MPaa!=0
                                                     ) ** 1.204,
                                rsb_m3m3
                                )
                        )
    return rs_m3m3


def __unf_rs_Velarde_m3m3__(p_MPaa=1, pb_MPaa=10, rsb_m3m3=100., gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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
        
        a1 = A[0] * gamma_gas ** A[1] * api ** A[2] * t_F ** A[3] * pb_psig ** A[4]
        a2 = B[0] * gamma_gas ** B[1] * api ** B[2] * t_F ** B[3] * pb_psig ** B[4]
        a3 = C[0] * gamma_gas ** C[1] * api ** C[2] * t_F ** C[3] * pb_psig ** C[4]
        pr = uc.MPa_2_psig(p_MPaa)  / pb_psig
        rsr = a1 * pr ** a2 + (1 - a1) * pr ** a3
        rs_m3m3 = rsr * rsb_m3m3
    else:
        rs_m3m3 = rsb_m3m3
    return rs_m3m3

# наивная векторизация
_unf_rs_Velarde_m3m3_ = np.vectorize(__unf_rs_Velarde_m3m3__)


def unf_rs_Velarde_m3m3(p_MPaa=1, pb_MPaa=10, rsb_m3m3=100., gamma_oil=0.86, gamma_gas=0.6, t_K=350):
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

    pr = np.where(pb_MPaa > 0.101314, 
                  uc.MPa_2_psig(p_MPaa)/(uc.MPa_2_psig(pb_MPaa)), 
                  0
                  )

    _go_ = -0.929328621908127 + 1 / gamma_oil
    _t_ = 0.0039158526769204 * t_K  - 1
    _pb_ = pb_MPaa - 0.101352932209575

    a1 = (0.0849029848623362 * gamma_gas**1.672608 * _go_**0.92987 *_t_**0.247235 * _pb_**1.056052) 
    a2 = (1.20743882814017 * _go_ ** 0.337711 * _t_** 0.132795 * _pb_**0.302065) / (gamma_gas ** 1.00475)
    a3 = 0.231607087371213 * _pb_ ** 0.047094 / ( gamma_gas**1.48548 * _go_**0.164741 * _t_**0.09133)

    rs_m3m3 = np.where(pr <= 0, 
                       0.0,
                       np.where( pr < 1, 
                                (a1 * pr ** a2 + (1 - a1) * pr ** a3) * rsb_m3m3,
                                rsb_m3m3
                                ) 
                        )
    return rs_m3m3


def __unf_rsb_Mccain_m3m3__(rsp_m3m3=10, gamma_oil=0.86, psp_MPaa=0.0, tsp_K=0.0):
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

# наивная векторизация
_unf_rsb_Mccain_m3m3_ = np.vectorize(__unf_rsb_Mccain_m3m3__)

def unf_rsb_Mccain_m3m3(rsp_m3m3=10, gamma_oil=0.86, psp_MPaa=0.0, tsp_K=0.0):
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
    api = uc.gamma_oil_2_api(gamma_oil)
    psp_psia = uc.MPa_2_psi(psp_MPaa)
    tsp_F = uc.K_2_F(tsp_K)

    nplogpsp_psia = np.log(psp_psia)
    nplogapi = np.log(api)
    z1 = -8.005 + 2.7 * nplogpsp_psia - 0.161 * nplogpsp_psia ** 2
    z2 = 1.224 - 0.5 * np.log(tsp_F)
    z3 = -1.587 + 0.0441 * nplogapi - 2.29e-5 * nplogapi ** 2
    z = z1 + z2 + z3
    rst_scfstb = np.exp(3.955 + 0.83 * z - 0.024 * z ** 2 + 0.075 * z ** 3)
    
    rsb = np.where(psp_MPaa * tsp_K > 0, 
                   rsp_scfstb + rst_scfstb,
                   np.where(rsp_m3m3 >= 0, 
                            1.1618 * rsp_scfstb,
                            0
                            )
                   )

    return uc.scfstb_2_m3m3(rsb)


def __unf_bo_above_pb_m3m3__(bob_m3m3=1.2, compr_o_1MPa=3e-3, pb_MPaa=10, p_MPaa=1):
    """
        Oil Formation Volume Factor according equation for pressure above bubble point pressure

    :param bob_m3m3: formation volume factor at bubble point pressure, m3m3
    :param compr_o_1MPa: weighted-average oil compressibility from bubblepoint pressure to a higher pressure of interest,1/MPa
    :param pb_MPaa: bubble point pressure, MPa
    :param p_MPaa: pressure, MPa
    :return: formation volume factor bo,m3m3

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011

    ! Actually, this correlation is belonged ro Vasquez & Beggs (1980). In some sources is
    noted that this is Standing correlation.

    ref2 Vazquez, M. and Beggs, H.D. 1980. Correlations for Fluid Physical Property Prediction.
    J Pet Technol 32 (6): 968-970. SPE-6719-PA

    """

    if p_MPaa <= pb_MPaa:
        bo = bob_m3m3
    else:
        bo = bob_m3m3 * np.exp(compr_o_1MPa * (pb_MPaa - p_MPaa))
    return bo

# наивная векторизация
_unf_bo_above_pb_m3m3_ = np.vectorize(__unf_bo_above_pb_m3m3__)


def unf_bo_above_pb_m3m3(bob_m3m3=1.2, compr_o_1MPa=3e-3, pb_MPaa=10, p_MPaa=1):
    """
        Oil Formation Volume Factor according equation for pressure above bubble point pressure

    :param bob_m3m3: formation volume factor at bubble point pressure, m3m3
    :param compr_o_1MPa: weighted-average oil compressibility from bubblepoint pressure to a higher pressure of interest,1/MPa
    :param pb_MPaa: bubble point pressure, MPa
    :param p_MPaa: pressure, MPa
    :return: formation volume factor bo,m3m3

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011

    ! Actually, this correlation is belonged ro Vasquez & Beggs (1980). In some sources is
    noted that this is Standing correlation.

    ref2 Vazquez, M. and Beggs, H.D. 1980. Correlations for Fluid Physical Property Prediction.
    J Pet Technol 32 (6): 968-970. SPE-6719-PA

    """

    return np.where(p_MPaa <= pb_MPaa, bob_m3m3, bob_m3m3 * np.exp(compr_o_1MPa * (pb_MPaa - p_MPaa)))


def _unf_bo_below_pb_m3m3_(rho_oil_st_kgm3=820, rs_m3m3=100, rho_oil_insitu_kgm3=700, gamma_gas=0.8):
    """
        Oil Formation Volume Factor according McCain correlation for pressure below bubble point pressure

    :param rho_oil_st_kgm3: density of stock-tank oil, kgm3
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param rho_oil_insitu_kgm3: Oil density at reservoir conditions, kgm3
    :param gamma_gas: specific gas  density(by air)
    :return: formation volume factor bo, m3m3

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011
    """

    rho_oil_st_lbft3 = uc.kgm3_2_lbft3(rho_oil_st_kgm3)
    rho_oil_insitu_lbft3 = uc.kgm3_2_lbft3(rho_oil_insitu_kgm3)
    rs_scfstb = uc.m3m3_2_scfstb(rs_m3m3)
    bo = (rho_oil_st_lbft3 + 0.01357 * rs_scfstb * gamma_gas) / rho_oil_insitu_lbft3
    return bo


def unf_bo_below_pb_m3m3(rho_oil_st_kgm3=820, rs_m3m3=100, rho_oil_insitu_kgm3=700, gamma_gas=0.8):
    """
        Oil Formation Volume Factor according McCain correlation for pressure below bubble point pressure

    :param rho_oil_st_kgm3: density of stock-tank oil, kgm3
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param rho_oil_insitu_kgm3: Oil density at reservoir conditions, kgm3
    :param gamma_gas: specific gas  density(by air)
    :return: formation volume factor bo, m3m3

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011
    """
    # коэффициенты преобразованы - смотри описанию в ноутбуке
    bo = (rho_oil_st_kgm3 + 1.22044505587208 * rs_m3m3 * gamma_gas) / rho_oil_insitu_kgm3
    return bo

def unf_bo_saturated_Standing_m3m3(rs_m3m3=100, gamma_gas=0.8, gamma_oil=0.86, t_K=300):
    """
        Oil Formation Volume Factor according Standing equation at bubble point pressure

    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param gamma_gas: specific gas density (by air)
    :param gamma_oil: specific oil density (by water)
    :param t_K: temperature, K
    :return: formation volume factor at bubble point pressure bo,m3m3

    ref1 Volumetric and phase behavior of oil field hydrocarbon systems / M.B. Standing Standing, M. B. 1981
    """

    rs_scfstb = uc.m3m3_2_scfstb(rs_m3m3)
    t_F = uc.K_2_F(t_K)
    bo = 0.972 + 1.47 * 10 ** (-4) * (rs_scfstb * (gamma_gas / gamma_oil) ** 0.5 + 1.25 * t_F) ** 1.175
    return bo


def __unf_density_oil_Mccain__(p_MPaa=1, pb_MPaa=10, co_1MPa=3e-3, rs_m3m3=10, gamma_gas=0.86, t_K=300, gamma_oil=0.86, gamma_gassp = 0):
    """
        Oil density according Standing, M.B., 1977; Witte, T.W., Jr., 1987; and McCain, W.D., Jr. and Hill, N.C.,
    1995 correlation.

    :param p_MPaa: pressure, MPaa
    :param pb_MPaa: bubble point pressure, MPaa
    :param co_1MPa: coefficient of isothermal compressibility, 1/MPa
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param gamma_gas: specific gas density (by air)
    :param t_K: temperature, K
    :param gamma_oil: specific oil density (by water)
    :param gamma_gassp: specific gas density in separator(by air)
    :return: oil density,kg/m3

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011
    """
    rs_scfstb = uc.m3m3_2_scfstb(rs_m3m3)
    t_F = uc.K_2_F(t_K)
    p_psia = uc.MPa_2_psi(p_MPaa)

    if gamma_gassp == 0:  # если нет данных о плотности газа в сепараторе
        gamma_gassp = gamma_gas
    def ro_po_equation(ro_po, vars):
        gamma_gassp, gamma_gas, gamma_oil, rs_scfstb = vars
        ro_a = -49.8930 + 85.0149 * gamma_gassp - 3.70373 * gamma_gassp * ro_po +\
            0.0479818 * gamma_gassp * ro_po ** 2 + 2.98914 * ro_po - 0.0356888 * ro_po ** 2
        return (rs_scfstb * gamma_gas + 4600 * gamma_oil) / (73.71 + rs_scfstb * gamma_gas / ro_a) -ro_po
    
    ro_po = 52.8 - 0.01 * rs_scfstb  # первое приближение
    ro_po = opt.fsolve(ro_po_equation, ro_po, [gamma_gassp, gamma_gas, gamma_oil, rs_scfstb])[0]
    
    if p_MPaa < pb_MPaa:
        p = p_psia 
    else:
        pb_psia = uc.MPa_2_psi(pb_MPaa)
        p = pb_psia

    dro_p = ((0.167 + 16.181 * 10 ** (-0.0425 * ro_po)) * (p / 1000) - 0.01 * 
             (0.299 + 263 * 10 ** (-0.0603 * ro_po)) * (p / 1000) ** 2)
    ro_bs = ro_po + dro_p
    dro_t = ((0.00302 + 1.505 * ro_bs ** (-0.951)) * (t_F - 60) ** 0.938 - 
             (0.0216 - 0.0233 * 10 ** (-0.0161 * ro_bs)) * (t_F - 60) ** 0.475)
    ro_or = ro_bs - dro_t

    if p_MPaa >= pb_MPaa:
        co_1psi = uc.compr_1pa_2_1psi(co_1MPa / 10 ** 6)
        ro_or = ro_or * np.exp(co_1psi * (p_psia - pb_psia))
    ro_or = uc.lbft3_2_kgm3(ro_or)
    return ro_or

# векторная версия для единообразия
_unf_density_oil_Mccain_ = np.vectorize(__unf_density_oil_Mccain__)

# векторная версия 
#TODO надо бы векторизовать быстрее, но не очень понятно пока как сделать
unf_density_oil_Mccain = np.vectorize(__unf_density_oil_Mccain__)

# старая версия с решением уравнения из VBA
# оставлено для справки
def ___unf_density_oil_Mccain___(p_MPaa=1, pb_MPaa=10, co_1MPa=3e-3, rs_m3m3=10, gamma_gas=0.8, t_K=300, gamma_oil=0.86, gamma_gassp = 0):
    """
        Oil density according Standing, M.B., 1977; Witte, T.W., Jr., 1987; and McCain, W.D., Jr. and Hill, N.C.,
    1995 correlation.

    :param p_MPaa: pressure, MPaa
    :param pb_MPaa: bubble point pressure, MPaa
    :param co_1MPa: coefficient of isothermal compressibility, 1/MPa
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param gamma_gas: specific gas density (by air)
    :param t_K: temperature, K
    :param gamma_oil: specific oil density (by water)
    :param gamma_gassp: specific gas density in separator(by air)
    :return: oil density,kg/m3

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011
    """

    rs_scfstb = uc.m3m3_2_scfstb(rs_m3m3)
    ro_po = 52.8 - 0.01 * rs_scfstb  # первое приближение
    if gamma_gassp == 0:  # если нет данных о плотности газа в сепараторе
        gamma_gassp = gamma_gas
    epsilon = 0.000001
    maxiter = 100  # максимальное число иттераций
    counter = 0
    ro_po_current = ro_po
    ro_po_previous = 0
    # TODO в идеале нужно также как для z-фактора сделать через scipy а не через цикл
    while abs(ro_po_current - ro_po_previous) > epsilon and counter < maxiter:
        ro_po_previous = ro_po_current
        ro_a = -49.8930 + 85.0149 * gamma_gassp - 3.70373 * gamma_gassp * ro_po +\
            0.0479818 * gamma_gassp * ro_po ** 2 + 2.98914 * ro_po - 0.0356888 * ro_po ** 2
        ro_po = (rs_scfstb * gamma_gas + 4600 * gamma_oil) / (73.71 + rs_scfstb * gamma_gas / ro_a)
        ro_po_current = ro_po
        counter = counter + 1
    p_psia = uc.Pa_2_psi(p_MPaa * 10 ** 6)
    t_F = uc.K_2_F(t_K)
    if p_MPaa < pb_MPaa:
        dro_p = (0.167 + 16.181 * 10 ** (-0.0425 * ro_po)) * (p_psia / 1000) - 0.01 * (
                    0.299 + 263 * 10 ** (-0.0603 * ro_po)) * (p_psia / 1000) ** 2
        ro_bs = ro_po + dro_p
        dro_t = (0.00302 + 1.505 * ro_bs ** (-0.951)) * (t_F - 60) ** 0.938 - (0.0216 - 0.0233 * 10 ** (-0.0161 * ro_bs)
                                                                               ) * (t_F - 60) ** 0.475
        ro_or = ro_bs - dro_t
    else:
        pb_psia = uc.Pa_2_psi(pb_MPaa * 10 ** 6)
        dro_p = (0.167 + 16.181 * 10 ** (-0.0425 * ro_po)) * (pb_psia / 1000) - 0.01 * (
                    0.299 + 263 * 10 ** (-0.0603 * ro_po)) * (pb_psia / 1000) ** 2
        ro_bs = ro_po + dro_p
        dro_t = (0.00302 + 1.505 * ro_bs ** (-0.951)) * (t_F - 60) ** 0.938 - (
                    0.0216 - 0.0233 * 10 ** (-0.0161 * ro_bs)) * (t_F - 60) ** 0.475
        ro_orb = ro_bs - dro_t
        co_1psi = uc.compr_1pa_2_1psi(co_1MPa / 10 ** 6)
        ro_or = ro_orb * np.exp(co_1psi * (p_psia - pb_psia))
    ro_or = uc.lbft3_2_kgm3(ro_or)
    return ro_or

    
def unf_density_oil_Standing(p_MPaa=1, pb_MPaa=10, co_1MPa=3e-3, rs_m3m3=10, bo_m3m3=1.1, gamma_gas=0.8, gamma_oil=0.86):
    """
        Oil density according Standing correlation.

    :param p_MPaa: pressure, MPaa
    :param pb_MPaa: bubble point pressure, MPaa
    :param co_1MPa: coefficient of isothermal compressibility, 1/MPa
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param bo_m3m3: oil formation volume factor, m3m3
    :param gamma_gas: specific gas density (by air)
    :param gamma_oil: specific oil density (by water)
    :return: oil density,kg/m3

    ref1 book Brill 2006, Production Optimization Using Nodal Analysis
    """
    po = (1000 * gamma_oil + 1.224 * gamma_gas * rs_m3m3) / bo_m3m3

    # TODO возможно это надо оставить, при давлении выше P нас, уже есть в fvf_oil степень с co
        
    return np.where(p_MPaa > pb_MPaa, po * np.exp(co_1MPa * (p_MPaa - pb_MPaa)), po)


def unf_compressibility_saturated_oil_McCain_1Mpa(p_MPa=1, pb_MPa=10, t_K=300, gamma_oil=0.86, rsb_m3m3=100):
    """
        Oil compressibility below bubble point (saturated oil)

    :param p_mpa:
    :param pb_mpa:
    :param t_k:
    :param gamma_oil:
    :param rsb_m3m3:
    :return:

    ref1 https://www.researchgate.net/publication/
    254529353_The_Oil_Compressibility_Below_Bubble_Point_Pressure_Revisited_-_Formulations_and_Estimations

    ref2 https://www.onepetro.org/download/journal-paper/SPE-15664-PA?id=journal-paper%2FSPE-15664-PA
    """
    rsb_scfstb = uc.m3m3_2_scfstb(rsb_m3m3)
    t_F = uc.K_2_F(t_K)
    api = uc.gamma_oil_2_api(gamma_oil)
    p_psia = uc.MPa_2_psi(p_MPa)
    pb_psia = uc.MPa_2_psi(pb_MPa)
    co_1psi = np.exp(-7.573 - 1.450 * np.log(p_psia) - 0.383 * np.log(pb_psia) + 1.402 * np.log(t_F) +
                     0.256 * np.log(api) + 0.449 * np.log(rsb_scfstb))
    co_1MPa = uc.compr_1psi_2_1MPa(co_1psi)# * 10 ** 5 # TODO надо разобраться и исправить
    return co_1MPa


def unf_compressibility_oil_VB_1Mpa(rs_m3m3, t_K, gamma_oil, p_MPaa, gamma_gas=0.6): # TODO above bubble point!!
    """
        Oil compressibility according to Vasquez & Beggs (1980) correlation

    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param t_K: temperature, K
    :param gamma_oil: specific oil density (by water)
    :param p_MPaa: pressure, MPaa
    :param gamma_gas: specific gas density (by air)
    :return: coefficient of isothermal compressibility co_1MPa, 1/MPa

    ref1 Vazquez, M. and Beggs, H.D. 1980. Correlations for Fluid Physical Property Prediction.
    J Pet Technol 32 (6): 968-970. SPE-6719-PA

    """

    if p_MPaa > 0:
        rs_scfstb = uc.m3m3_2_scfstb(rs_m3m3)
        t_F = uc.K_2_F(t_K)
        api = uc.gamma_oil_2_api(gamma_oil)
        p_psia = uc.MPa_2_psi(p_MPaa)
        co_1MPa = uc.compr_1psi_2_1MPa( (-1433 + 5 * rs_scfstb + 17.2 * t_F - 1180 * gamma_gas + 12.61 * api) / (1e5 * p_psia))
    else:
        co_1MPa = 0
    return co_1MPa

def unf_compessibility_oil_VB_1MPa_vba(rsb_m3m3, t_k, gamma_oil, gamma_gas):
    co_1atm = (28.1 * rsb_m3m3 + 30.6 * t_k - 1180 * gamma_gas + 1784 / gamma_oil - 10910)
    return 1/uc.atm_2_bar(1/co_1atm)/10  #*))) **когда не дружишь с математикой ***когда слишком много времени потратил на uniflocpy

def unf_gamma_gas_Mccain(rsp_m3m3, rst_m3m3, gamma_gassp=0.8, gamma_oil=0.86, psp_MPaa=0, tsp_K=0):
    """
        Correlation for weighted-average specific gravities of surface gases

    :param rsp_m3m3: separator producing gas-oil ratio, m3m3
    :param rst_m3m3: stock-tank producing gas-oil ratio, m3m3
    :param gamma_gassp:  separator gas specific gravity
    :param gamma_oil: specific oil density(by water)
    :param psp_MPaa: pressure in separator, MPaa
    :param tsp_K: temperature in separator, K
    :return: weighted-average specific gravities of surface gases

    часто условия в сепараторе неизвестны, может считать и без них по приблизительной формуле

    ref1 "Reservoir oil bubblepoint pressures revisited; solution gas–oil ratios and surface gas specific gravities",
    J. VELARDE, W.D. MCCAIN, 2002

    """

    api = uc.gamma_oil_2_api(gamma_oil)
    psp_psia = uc.MPa_2_psi(psp_MPaa)
    tsp_F = uc.K_2_F(tsp_K)
    rsp_scfstb = uc.m3m3_2_scfstb(rsp_m3m3)
    rst_scfstb = uc.m3m3_2_scfstb(rst_m3m3)

    nplogpsp_psia = np.log(psp_psia) 
    nplogrsp_scfstb = np.log(rsp_scfstb)

    z1 = -17.275 + 7.9597 * nplogpsp_psia - 1.1013 * nplogpsp_psia ** 2 + 2.7735e-2 \
        * nplogpsp_psia ** 3 + 3.2287e-3 * nplogpsp_psia ** 4
    z2 = -0.3354 - 0.3346 * nplogrsp_scfstb + 0.1956 * nplogrsp_scfstb ** 2 - 3.4374e-2 \
        * nplogrsp_scfstb ** 3 + 2.08e-3 * nplogrsp_scfstb ** 4
    z3 = 3.705 - 0.4273 * api + 1.818e-2 * api ** 2 - 3.459e-4 \
        * api ** 3 + 2.505e-6 * api ** 4
    z4 = -155.52 + 629.61 * gamma_gassp - 957.38 * gamma_gassp ** 2 + 647.57 \
        * gamma_gassp ** 3 - 163.26 * gamma_gassp ** 4
    z5 = 2.085 - 7.097e-2 * tsp_F + 9.859e-4 * tsp_F ** 2 \
        - 6.312e-6 * tsp_F ** 3 + 1.4e-8 * tsp_F ** 4
    z = z1 + z2 + z3 + z4 + z5
    # Stock-tank gas specific gravity
    gamma_gasst = 1.219 + 0.198 * z + 0.0845 * z ** 2 + 0.03 * z ** 3 + 0.003 * z ** 4
      
    return np.where(psp_MPaa * rsp_m3m3 * rst_m3m3 * gamma_gassp > 0,
                    (gamma_gassp * rsp_scfstb + gamma_gasst * rst_scfstb) / (rsp_scfstb + rst_scfstb),
                    np.where(gamma_gassp >= 0,
                            1.066 * gamma_gassp,
                            0
                            )
                    )
