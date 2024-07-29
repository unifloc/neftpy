from neftpy.uconvert import *
from neftpy.uconst import *

import numpy as np

""" 
====================================================================================================
Корреляции расчета давления насыщения
====================================================================================================
"""

def unf_pb_Standing_MPaa(rsb_m3m3:float=100, 
                         gamma_oil:float=0.86, 
                         gamma_gas:float=0.6, 
                         t_K:float=350
                         )->float:
    """
    Расчет давления насыщения Standing (1947)

    :param rsb_m3m3: газосодержание при давлении насыщения, должно быть указано, м3/м3 
    :param gamma_oil: удельная плотность нефти (от воды) 
    :param gamma_gas: удельная плотность газа (от воздуха) 
    :param t_K: температура, К 
    :return: давление насыщения, МПа абсолютное 

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
        pb_MPaa = (pb_MPaa - P_SC_MPa) * rsb_old / min_rsb + P_SC_MPa  
    return pb_MPaa


def unf_pb_Valko_MPaa(rsb_m3m3:float=100, 
                        gamma_oil:float=0.86, 
                        gamma_gas:float=0.6,
                        t_K:float=350
                        )->float:
    """
    Расчет давления насыщения Valko McCain (2002)

    :param rsb_m3m3: газосодержание при давлении насыщения, должно быть указано, м3/м3 
    :param gamma_oil: удельная плотность нефти (от воды) 
    :param gamma_gas: удельная плотность газа (от воздуха) 
    :param t_K: температура, К 
    :return: давление насыщения, МПа абсолютное 

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
    корреляция Valko получена с использованием непараметрической регресии GRACE метод
    особенность подхода - за пределеми интервала адаптации ассимптотики не соблюдаются
    поэтому их устанавливаем вручную
    для больших значений газосодержания продолжим линейный тренд корреляции
    """
    if rsb_old < min_rsb:
        pb_MPaa = (pb_MPaa - 0.1013) * rsb_old / min_rsb + 0.1013
    if rsb_old > max_rsb:
        pb_MPaa = (pb_MPaa - 0.1013) * rsb_old / max_rsb + 0.1013

    return pb_MPaa


""" 
====================================================================================================
Корреляции расчета газосодержания (Gas Solution Ratio)
====================================================================================================
"""


def unf_rs_Standing_m3m3(p_MPaa:float=1, 
                         pb_MPaa:float=10,
                         rsb_m3m3:float=0, 
                         gamma_oil:float=0.86, 
                         gamma_gas:float=0.6, 
                         t_K:float=350
                         )->float:
    """
    Расчет газосодержания в нефти при заданном давлении и температуре Standing (1947)
    используется зависимость обратная к корреляции между давлением насыщения и газосодержанием

    :param p_MPaa: давление, MPa
    :param pb_MPaa: давление насыщения, MPa
    :param rsb_m3m3: газосодержание при давлении насыщения, m3/m3
    :param gamma_oil: удельная плотность нефти
    :param gamma_gas: удельная плотность газа
    :param t_K: температура, К
    :return: газосодержание при заданном давлении и температуре, m3/m3

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



def unf_rs_Velarde_m3m3(p_MPaa:float=1, 
                        pb_MPaa:float=10, 
                        rsb_m3m3:float=100., 
                        gamma_oil:float=0.86, 
                        gamma_gas:float=0.6, 
                        t_K:float=350
                        )->float:
    """
    газосодержание по Velarde McCain (1999) 

    :param p_MPaa: давление, MPa
    :param pb_MPaa: давление насыщения, MPa
    :param rsb_m3m3: газосодержание при давлении насыщения, m3/m3
    :param gamma_oil: удельная плотность нефти
    :param gamma_gas: удельная плотность газа
    :param t_K: температура, К
    :return: газосодержание при заданном давлении и температуре, m3/m3

    ref1 "Correlation of Black Oil Properties at Pressures Below Bubblepoint Pressure—A New Approach",
    J. VELARDE, T.A. BLASINGAME Texas A&M University, W.D. MCCAIN, JR. S.A. Holditch & Associates, Inc 1999

    """

    api = gamma_oil_2_api(gamma_oil)
    t_F = K_2_F(t_K)
    pb_psig = MPa_2_psi(pb_MPaa) - P_SC_PSI

    if pb_psig > 0:
        pr = MPa_2_psig(p_MPaa)  / pb_psig
    else:
        pr = 0

    if pr <= 0:
        rs_m3m3 = 0.0
    elif pr < 1:
        A = (9.73e-7, 1.672608, 0.929870, 0.247235, 1.056052)
        B = (0.022339, -1.004750, 0.337711, 0.132795, 0.302065)
        C = (0.725167, -1.485480, -0.164741, -0.091330, 0.047094)
        
        a1 = A[0] * gamma_gas ** A[1] * api ** A[2] * t_F ** A[3] * pb_psig ** A[4]
        a2 = B[0] * gamma_gas ** B[1] * api ** B[2] * t_F ** B[3] * pb_psig ** B[4]
        a3 = C[0] * gamma_gas ** C[1] * api ** C[2] * t_F ** C[3] * pb_psig ** C[4]
        pr = MPa_2_psig(p_MPaa)  / pb_psig
        rsr = a1 * pr ** a2 + (1 - a1) * pr ** a3
        rs_m3m3 = rsr * rsb_m3m3
    else:
        rs_m3m3 = rsb_m3m3
    return rs_m3m3



def unf_rsb_Mccain_m3m3(rsp_m3m3:float=10, 
                            gamma_oil:float=0.86, 
                            psp_MPaa:float=0.0, 
                            tsp_K:float=0.0
                            )->float:
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

    rsp_scfstb = m3m3_2_scfstb(rsp_m3m3)
    if psp_MPaa > 0 and tsp_K > 0:
        api = gamma_oil_2_api(gamma_oil)
        psp_psia = Pa_2_psi(psp_MPaa * 1e6)
        tsp_F = K_2_F(tsp_K)
        z1 = -8.005 + 2.7 * np.log(psp_psia) - 0.161 * np.log(psp_psia)**2
        z2 = 1.224 - 0.5 * np.log(tsp_F)
        z3 = -1.587 + 0.0441 * np.log(api) - 2.29e-5 * np.log(api)**2
        z = z1 + z2 + z3
        rst_scfstb = np.exp(3.955 + 0.83*z - 0.024 * z**2 + 0.075 * z**3)
        rsb = rsp_scfstb + rst_scfstb
    elif rsp_m3m3 >= 0:
        rsb = 1.1618 * rsp_scfstb
    else:
        rsb = 0
    rsb = scfstb_2_m3m3(rsb)
    return rsb


