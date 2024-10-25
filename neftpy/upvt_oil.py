from neftpy.uconvert import *
from neftpy.uconst import *

import numpy as np
import scipy.optimize as opt

""" 
====================================================================================================
Расчет давления насыщения
====================================================================================================
"""

def unf_pb_Standing_MPaa(t_K:float=350,
                         rsb_m3m3:float=100, 
                         gamma_oil:float=0.86, 
                         gamma_gas:float=0.6
                         )->float:
    """
    Расчет давления насыщения Standing (1947)

    :param t_K: температура, К 
    :param rsb_m3m3: газосодержание при давлении насыщения, должно быть указано, м3/м3 
    :param gamma_oil: удельная плотность нефти (от воды) 
    :param gamma_gas: удельная плотность газа (от воздуха) 
    :return: давление насыщения, МПа абсолютное 

    ref1 "A Pressure-Volume-Temperature Correlation for Mixtures of California Oil and Gases",
    M.B. Standing, Drill. & Prod. Prac., API, 1947.

    ref2  "Стандарт компании Юкос. Физические свойства нефти. Методы расчета." Афанасьев В.Ю., Хасанов М.М. и др. 2002 г
    """


    min_rsb = 1.8
    rsb_old = np.copy(rsb_m3m3)
    rsb_m3m3 = np.where(rsb_m3m3 < min_rsb, min_rsb, rsb_m3m3)
    yg = RHO_AIR_kgm3 + 0.001648 * t_K - 1.769 / gamma_oil         # gas molar fraction
    pb_MPaa = 0.5197 * (rsb_m3m3 / gamma_gas) ** 0.83 * 10 ** yg
    # for low rsb values, we set the asymptotics Pb = 1 atma at Rsb = 0
    # for large rsb values do not correct what the correlation gives
    return np.where(rsb_old < min_rsb, 
                    (pb_MPaa - P_SC_MPa) * rsb_old / min_rsb + P_SC_MPa, 
                    pb_MPaa
                    )


def unf_pb_Valko_MPaa(t_K:float=350,
                      rsb_m3m3:float=100, 
                      gamma_oil:float=0.86, 
                      gamma_gas:float=0.6
                      )->float:
    """
    Расчет давления насыщения Valko McCain (2002)

    :param t_K: температура, К 
    :param rsb_m3m3: газосодержание при давлении насыщения, должно быть указано, м3/м3 
    :param gamma_oil: удельная плотность нефти (от воды) 
    :param gamma_gas: удельная плотность газа (от воздуха) 
    :return: давление насыщения, МПа абсолютное 

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
    для низких значений газосодержания зададим асимптотику Pb = 1 атма при Rsb = 0
    корреляция Valko получена с использованием непараметрической регресии GRACE метод
    особенность подхода - за пределеми интервала адаптации ассимптотики не соблюдаются
    поэтому их устанавливаем вручную
    для больших значений газосодержания продолжим линейный тренд корреляции
    """

    pb_MPaa = np.where(rsb_old < min_rsb, 
                       (pb_MPaa - 0.1013) * rsb_old / min_rsb + 0.1013, 
                       np.where(rsb_old > max_rsb, 
                                (pb_MPaa - 0.1013) * rsb_old / max_rsb + 0.1013, 
                                pb_MPaa)
                        )

    return pb_MPaa


def unf_pb_Glaso_MPaa(t_K:float=350, 
                      rsb_m3m3:float=100, 
                      gamma_oil:float=0.85, 
                      gamma_gas:float=0.8
                      )->float:
    """
        Glaso correlation(1980) for bubble point pressure

    :param t_K: temperature in K
    :param rsb_m3m3: gas-oil ratio in m3/m3
    :param gamma_oil: oil density (by water)
    :param gamma_gas: gas density (by air)
    :return: bubble point pressure im MPa abs

    ref Generalized Pressure-Volume-Temperature Correlations, Glaso, 1980
    """

    #  можно дополнить код, поправками на неуглеводородные составляющие в нефти, в статье есть
    api = gamma_oil_2_api(gamma_oil)
    t_F = K_2_F(t_K)
    rs_scfstb = m3m3_2_scfstb(rsb_m3m3)
    pb = (rs_scfstb / gamma_gas) ** 0.816 * (t_F ** 0.172 / api ** 0.989)
    log_pb = np.log10(pb)
    pb = psi_2_MPa(10 ** (1.7669 + 1.7447 * log_pb - 0.30218 * log_pb ** 2)) 
    return pb


""" 
====================================================================================================
Расчет газосодержания (Gas Solution Ratio) в нефти
====================================================================================================
"""


def unf_rs_Standing_m3m3(p_MPaa:float=1, 
                         t_K:float=350,
                         pb_MPaa:float=10,
                         rsb_m3m3:float=0, 
                         gamma_oil:float=0.86, 
                         gamma_gas:float=0.6
                         )->float:
    """
    Расчет газосодержания в нефти при заданном давлении и температуре Standing (1947)
    используется зависимость обратная к корреляции между давлением насыщения и газосодержанием

    :param p_MPaa: давление, MPa
    :param t_K: температура, К
    :param pb_MPaa: давление насыщения, MPa
    :param rsb_m3m3: газосодержание при давлении насыщения, m3/m3
    :param gamma_oil: удельная плотность нефти
    :param gamma_gas: удельная плотность газа
    :return: газосодержание при заданном давлении и температуре, m3/m3
             или производная - в зависимости от флага calc_drs_dp

    ref1 "A Pressure-Volume-Temperature Correlation for Mixtures of California Oil and Gases",
    M.B. Standing, Drill. & Prod. Prac., API, 1947.

    ref2  "Стандарт компании Юкос. Физические свойства нефти. Методы расчета." Афанасьев В.Ю., Хасанов М.М. и др. 2002 г
    может считать в случае если нет давления насыщения и газосодержания при давлении насыщения, корреляция не точная
    """

    yg = RHO_AIR_kgm3 + 0.001648 * t_K - 1.769 / gamma_oil

    rs_m3m3 = np.where(pb_MPaa * rsb_m3m3 == 0, 
                        gamma_gas * (1.92 * p_MPaa / 10 ** yg) ** 1.204, 
                        np.where(p_MPaa < pb_MPaa, 
                                rsb_m3m3 * np.divide(p_MPaa, pb_MPaa, 
                                                    where=pb_MPaa!=0
                                                    ) ** 1.204,
                                rsb_m3m3
                                )
                        )
    return rs_m3m3

def unf_drs_dp_Standing_m3m3(p_MPaa:float=1, 
                             t_K:float=350,
                             pb_MPaa:float=10,
                             rsb_m3m3:float=0, 
                             gamma_oil:float=0.86, 
                             gamma_gas:float=0.6
                             )->float:
    """
    Расчет производной от газосодержания в нефти по давлению Standing (1947)
    
    :param p_MPaa: давление, MPa
    :param t_K: температура, К
    :param pb_MPaa: давление насыщения, MPa
    :param rsb_m3m3: газосодержание при давлении насыщения, m3/m3
    :param gamma_oil: удельная плотность нефти
    :param gamma_gas: удельная плотность газа
    :return:  производная 

    ref1 "A Pressure-Volume-Temperature Correlation for Mixtures of California Oil and Gases",
    M.B. Standing, Drill. & Prod. Prac., API, 1947.

    ref2  "Стандарт компании Юкос. Физические свойства нефти. Методы расчета." Афанасьев В.Ю., Хасанов М.М. и др. 2002 г
    может считать в случае если нет давления насыщения и газосодержания при давлении насыщения, корреляция не точная
    """

    yg = RHO_AIR_kgm3 + 0.001648 * t_K - 1.769 / gamma_oil

    drs_dp = np.where(pb_MPaa * rsb_m3m3 == 0, 
                        gamma_gas * (1.92 / 10**yg) ** 1.204 * 1.204 * p_MPaa**0.204, 
                        np.where(p_MPaa < pb_MPaa, 
                                rsb_m3m3 * np.divide(1, pb_MPaa, 
                                                    where=pb_MPaa!=0
                                                    ) ** 1.204 * p_MPaa ** 0.204,
                                0
                                )
                        )
    return drs_dp

def _unf_rs_Velarde_m3m3_(p_MPaa:float=1, 
                          t_K:float=350,
                          pb_MPaa:float=10, 
                          rsb_m3m3:float=100., 
                          gamma_oil:float=0.86, 
                          gamma_gas:float=0.6, 
                          )->float:
    """
    газосодержание по Velarde McCain (1999) 

    :param p_MPaa: давление, MPa
    :param t_K: температура, К
    :param pb_MPaa: давление насыщения, MPa
    :param rsb_m3m3: газосодержание при давлении насыщения, m3/m3
    :param gamma_oil: удельная плотность нефти
    :param gamma_gas: удельная плотность газа
    :return: газосодержание при заданном давлении и температуре, m3/m3

    ref1 "Correlation of Black Oil Properties at Pressures Below Bubblepoint Pressure—A New Approach",
    J. VELARDE, T.A. BLASINGAME Texas A&M University, W.D. MCCAIN, JR. S.A. Holditch & Associates, Inc 1999

    """

    # на всякий случай тут старая не векторизованная реализация

        
    pb_estimation_Valko_McCain = unf_pb_Valko_MPaa(RS_MAX_Velarde, gamma_oil, gamma_gas, t_K)
    if (pb_MPaa > pb_estimation_Valko_McCain):
        if p_MPaa < pb_MPaa:
            return rsb_m3m3 * p_MPaa / pb_MPaa
        else:
            return rsb_m3m3
        
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


def unf_rs_Velarde_m3m3(p_MPaa:float=1, 
                        t_K:float=350,
                        pb_MPaa:float=10, 
                        rsb_m3m3:float=100., 
                        gamma_oil:float=0.86, 
                        gamma_gas:float=0.6, 
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
    pr = np.where(pb_MPaa > P_SC_MPa, 
                  MPa_2_psig(p_MPaa)/(MPa_2_psig(pb_MPaa)), 
                  0
                  )

    _go_ = -0.929328621908127 + 1 / gamma_oil
    _t_ = 0.0039158526769204 * t_K  - 1
    _pb_ = pb_MPaa - P_SC_MPa

    a1 = (0.0849029848623362 * gamma_gas**1.672608 * _go_**0.92987 *_t_**0.247235 * _pb_**1.056052) 
    a2 = (1.20743882814017 * _go_ ** 0.337711 * _t_** 0.132795 * _pb_**0.302065) / (gamma_gas ** 1.00475)
    a3 = 0.231607087371213 * _pb_ ** 0.047094 / ( gamma_gas**1.48548 * _go_**0.164741 * _t_**0.09133)

    
    pb_estimation_Valko_McCain = unf_pb_Valko_MPaa(RS_MAX_Velarde, gamma_oil, gamma_gas, t_K)
     
    rs_m3m3 = np.where(pb_MPaa > pb_estimation_Valko_McCain, 
                       np.where(p_MPaa < pb_MPaa,
                                rsb_m3m3 * np.divide(p_MPaa, pb_MPaa, where=pb_MPaa!=0),
                                rsb_m3m3),
                       np.where(pr <= 0, 
                                0.0,
                                np.where( pr < 1, 
                                            (a1 * pr ** a2 + (1 - a1) * pr ** a3) * rsb_m3m3,
                                            rsb_m3m3
                                            ) 
                                    )
                        )
    #TODO rnt можно добавить тут расчет производной - в vba версии есть

    return rs_m3m3


def unf_rsb_Mccain_m3m3(rsp_m3m3:float=10, 
                        psp_MPaa:float=0.0, 
                        tsp_K:float=0.0,
                        gamma_oil:float=0.86, 
                        )->float:
    """
        Solution Gas-oil ratio at bubble point pressure calculation according to McCain (2002) correlation
    taking into account the gas losses at separator and stock tank

    :param rsp_m3m3: separator producing gas-oil ratio, m3m3
    :param psp_MPaa: pressure in separator, MPaa
    :param tsp_K: temperature in separator, K
    :param gamma_oil: specific oil density(by water)
    :return: solution gas-oil ratio at bubble point pressure, rsb in m3/m3

    часто условия в сепараторе неизвестны, может считать и без них по приблизительной формуле

    ref1 "Reservoir oil bubblepoint pressures revisited; solution gas–oil ratios and surface gas specific gravities",
    J. VELARDE, W.D. MCCAIN, 2002
    """

    rsp_scfstb = m3m3_2_scfstb(rsp_m3m3)
    api = gamma_oil_2_api(gamma_oil)
    psp_psia = MPa_2_psi(psp_MPaa)
    tsp_F = K_2_F(tsp_K)

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

    return scfstb_2_m3m3(rsb)

""" 
====================================================================================================
Расчет объемного коэффициента нефти (Oil Formation Volume Factor FVF)
====================================================================================================
"""

def unf_bo_above_pb_m3m3(p_MPaa:float=1,
                         pb_MPaa:float=10, 
                         bob_m3m3:float=1.2,
                         compr_o_1MPa:float=3e-3, 
                         )->float:
    """
        Oil Formation Volume Factor according equation for pressure above bubble point pressure

    :param p_MPaa: pressure, MPa
    :param pb_MPaa: bubble point pressure, MPa
    :param bob_m3m3: formation volume factor at bubble point pressure, m3m3
    :param compr_o_1MPa: weighted-average oil compressibility from bubblepoint pressure to a higher pressure of interest,1/MPa
    :return: formation volume factor bo,m3m3

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011

    ! Actually, this correlation is belonged ro Vasquez & Beggs (1980). In some sources is
    noted that this is Standing correlation.

    ref2 Vazquez, M. and Beggs, H.D. 1980. Correlations for Fluid Physical Property Prediction.
    J Pet Technol 32 (6): 968-970. SPE-6719-PA

    """

    return np.where(p_MPaa <= pb_MPaa, 
                    bob_m3m3, 
                    bob_m3m3 * np.exp(compr_o_1MPa * (pb_MPaa - p_MPaa)))


def unf_bo_below_pb_m3m3(rho_oil_st_kgm3:float=820, 
                         rho_oil_insitu_kgm3:float=700, 
                         rs_m3m3:float=100, 
                         gamma_gas:float=0.8
                         )->float:
    """
        Oil Formation Volume Factor according McCain correlation for pressure below bubble point pressure

    :param rho_oil_st_kgm3: density of stock-tank oil, kgm3
    :param rho_oil_insitu_kgm3: Oil density at reservoir conditions, kgm3
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param gamma_gas: specific gas  density(by air)
    :return: formation volume factor bo, m3m3

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011
    """
    # коэффициенты преобразованы - смотри описанию в ноутбуке
    bo = (rho_oil_st_kgm3 + RHO_AIR_kgm3 * rs_m3m3 * gamma_gas) / rho_oil_insitu_kgm3
    return bo


def unf_bo_saturated_Standing_m3m3(t_K:float=300,
                                   rs_m3m3:float=100,
                                   gamma_oil:float=0.86, 
                                   gamma_gas:float=0.8 
                                   )->float:
    """
        Oil Formation Volume Factor according Standing equation at bubble point pressure

    :param t_K: temperature, K
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param gamma_oil: specific oil density (by water)
    :param gamma_gas: specific gas density (by air)
    :return: formation volume factor at bubble point pressure bo,m3m3

    ref1 Volumetric and phase behavior of oil field hydrocarbon systems / M.B. Standing Standing, M. B. 1981
    """

    rs_scfstb = m3m3_2_scfstb(rs_m3m3)
    t_F = K_2_F(t_K)
    return 0.972 + 1.47e-4 * (rs_scfstb * (gamma_gas / gamma_oil) ** 0.5 + 1.25 * t_F) ** 1.175


def unf_bo_saturated_Glaso_m3m3(t_K:float, 
                                rs_m3m3:float,
                                gamma_oil:float, 
                                gamma_gas:float
                                )->float:
    """
        Glaso correlation(1980) for formation volume factor at bubble point pressure

    :param rs_m3m3: gas-oil ratio in m3/m3
    :param t_K: temperature in K
    :param gamma_oil: oil density (by water)
    :param gamma_gas: gas density (by air)
    :return: formation volume factor at bubble point pressure in m3/m3

    ref Generalized Pressure-Volume-Temperature Correlations, Glaso, 1980
    """

    t_F = K_2_F(t_K)
    rs_scfstb = m3m3_2_scfstb(rs_m3m3)
    bob = rs_scfstb * (gamma_gas / gamma_oil) ** 0.526 + 0.968 * t_F
    bob = 10 ** (-6.58511 + 2.91329 * np.log10(bob) - 0.27683 * np.log10(bob) ** 2) + 1
    return bob


def unf_bo_below_Glaso_m3m3(p_MPaa:float, 
                            t_K:float, 
                            rs_m3m3:float,
                            gamma_oil:float, 
                            gamma_gas:float 
                            )->float:
    """
        Glaso correlation(1980) for total formation volume factor below bubble point pressure

    :param rs_m3m3: gas-oil ratio in m3/m3
    :param t_K: temperature in K
    :param gamma_oil: oil density (by water)
    :param gamma_gas: gas density (by air)
    :param p_MPaa: pressure in MPaa
    :return: total formation volume factor below bubble point pressure in m3/m3

    ref Generalized Pressure-Volume-Temperature Correlations, Glaso, 1980
    """

    t_F = K_2_F(t_K)
    rs_scfstb = m3m3_2_scfstb(rs_m3m3)
    p_psia = MPa_2_psi(p_MPaa)
    log_bt = np.log10( rs_scfstb * t_F ** 0.5 / gamma_gas ** 0.3 * gamma_oil ** (2.9 * 10 ** (-0.00027 * rs_scfstb)) * p_psia ** -1.1089)

    return 10 ** (8.0135e-2 + 4.7257e-1 * log_bt + 1.7351e-1 * log_bt ** 2)


""" 
====================================================================================================
Расчет плотности нефти
====================================================================================================
"""

def unf_rho_oil_Mccain_kgm3(p_MPaa:float=1,
                           t_K:float=300,  
                           pb_MPaa:float=10,
                           rs_m3m3:float=10,  
                           co_1MPa:float=3e-3, 
                           gamma_oil:float=0.86, 
                           gamma_gas:float=0.86, 
                           gamma_gassp:float = 0,
                           )->float:
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
    rs_scfstb = m3m3_2_scfstb(rs_m3m3)
    t_F = K_2_F(t_K)
    p_psia = MPa_2_psi(p_MPaa)

    gamma_gassp = np.where(gamma_gassp == 0, gamma_gas, gamma_gassp)

    def ro_po_equation(ro_po, vars):
        gamma_gassp, gamma_gas, gamma_oil, rs_scfstb = vars
        ro_a = -49.8930 + 85.0149 * gamma_gassp - 3.70373 * gamma_gassp * ro_po +\
            0.0479818 * gamma_gassp * ro_po ** 2 + 2.98914 * ro_po - 0.0356888 * ro_po ** 2
        return (rs_scfstb * gamma_gas + 4600 * gamma_oil) / (73.71 + rs_scfstb * gamma_gas / ro_a) -ro_po
    
    ro_po = 52.8 - 0.01 * rs_scfstb  # первое приближение
    ro_po = opt.fsolve(ro_po_equation, ro_po, [gamma_gassp, gamma_gas, gamma_oil, rs_scfstb])
    pb_psia = MPa_2_psi(pb_MPaa)
    p = np.where(p_MPaa < pb_MPaa, p_psia, pb_psia)

    dro_p = ((0.167 + 16.181 * 10 ** (-0.0425 * ro_po)) * (p / 1000) - 0.01 * 
             (0.299 + 263 * 10 ** (-0.0603 * ro_po)) * (p / 1000) ** 2)
    ro_bs = ro_po + dro_p
    dro_t = ((0.00302 + 1.505 * ro_bs ** (-0.951)) * (t_F - 60) ** 0.938 - 
             (0.0216 - 0.0233 * 10 ** (-0.0161 * ro_bs)) * (t_F - 60) ** 0.475)
    ro_or = ro_bs - dro_t

    ro_or = np.where(p_MPaa >= pb_MPaa, 
                     ro_or * np.exp(compr_1pa_2_1psi(co_1MPa / 1e6) * (p_psia - pb_psia)),
                     ro_or)  
    return  lbft3_2_kgm3(ro_or)


def unf_rho_oil_Standing_kgm3(p_MPaa:float=1, 
                             rs_m3m3:float=10, 
                             pb_MPaa:float=10, 
                             bo_m3m3:float=1.1, 
                             co_1MPa:float=3e-3,
                             gamma_oil:float=0.86, 
                             gamma_gas:float=0.8
                             )->float:
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
    po = (RHO_WATER_SC_kgm3 * gamma_oil + RHO_AIR_kgm3 * gamma_gas * rs_m3m3) / bo_m3m3
        
    return np.where(p_MPaa > pb_MPaa, po * np.exp(co_1MPa * (p_MPaa - pb_MPaa)), po)


""" 
====================================================================================================
Расчет сжимаемости нефти
====================================================================================================
"""
def unf_compressibility_saturated_oil_McCain_1Mpa(p_MPa:float=1, 
                                                  t_K:float=300, 
                                                  pb_MPa:float=10, 
                                                  rsb_m3m3:float=100, 
                                                  gamma_oil:float=0.86
                                                  )->float:
    """
        Oil compressibility below bubble point (saturated oil)

    :param p_mpa: давление расчета
    :param pb_mpa: давление насыщения
    :param t_k: температура расчета
    :param gamma_oil: плотность нефти
    :param rsb_m3m3: газосодержание
    :return:

    ref1 https://www.researchgate.net/publication/
    254529353_The_Oil_Compressibility_Below_Bubble_Point_Pressure_Revisited_-_Formulations_and_Estimations

    ref2 https://www.onepetro.org/download/journal-paper/SPE-15664-PA?id=journal-paper%2FSPE-15664-PA
    """
    rsb_scfstb = m3m3_2_scfstb(rsb_m3m3)
    t_F = K_2_F(t_K)
    api = gamma_oil_2_api(gamma_oil)
    p_psia = MPa_2_psi(p_MPa)
    pb_psia = MPa_2_psi(pb_MPa)
    co_1psi = np.exp(-7.573 - 1.450 * np.log(p_psia) - 0.383 * np.log(pb_psia) + 1.402 * np.log(t_F) +
                     0.256 * np.log(api) + 0.449 * np.log(rsb_scfstb))
    co_1MPa = compr_1psi_2_1MPa(co_1psi)
    return co_1MPa
    #TODO в тесте большое расхожнение, надо бы сравнить с vba и разобраться


def unf_compressibility_oil_VB_1Mpa(p_MPaa:float, 
                                    t_K:float, 
                                    rs_m3m3:float=100,
                                    gamma_oil:float=0.86, 
                                    gamma_gas:float=0.6
                                    )->float: # TODO above bubble point!!
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

    rs_scfstb = m3m3_2_scfstb(rs_m3m3)
    t_F = K_2_F(t_K)
    api = gamma_oil_2_api(gamma_oil)
    p_psia = MPa_2_psi(p_MPaa)
    
    return np.where(p_MPaa > 0, 
                    compr_1psi_2_1MPa( (-1433 + 5 * rs_scfstb + 17.2 * t_F - 1180 * gamma_gas + 12.61 * api) / (1e5 * p_psia)),
                    0.0
                    )


def _unf_compessibility_oil_VB_1MPa_vba_(t_K:float, 
                                       rsb_m3m3:float,
                                       gamma_oil:float, 
                                       gamma_gas:float,
                                       )->float:
    co_1atm = (28.1 * rsb_m3m3 + 30.6 * t_K - 1180 * gamma_gas + 1784 / gamma_oil - 10910)
    return 1/atm_2_bar(1/co_1atm)/10  #*))) **когда не дружишь с математикой ***когда слишком много времени потратил на uniflocpy
    #TODO на сравнить с реализацией не vba и понять, что оставить


""" 
====================================================================================================
Расчет плотности газа на поверхности
====================================================================================================
"""
def unf_gamma_gas_Mccain(psp_MPaa:float, 
                         tsp_K:float,
                         rsp_m3m3:float, 
                         rst_m3m3:float,  
                         gamma_oil:float=0.86, 
                         gamma_gassp:float=0.8,
                         )->float:
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

    api = gamma_oil_2_api(gamma_oil)
    psp_psia = MPa_2_psi(psp_MPaa)
    tsp_F = K_2_F(tsp_K)
    rsp_scfstb = m3m3_2_scfstb(rsp_m3m3)
    rst_scfstb = m3m3_2_scfstb(rst_m3m3)

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


def unf_McCain_specificgravity(p_MPaa:float, 
                               t_K:float, 
                               rsb_m3m3:float, 
                               gamma_oil:float, 
                               gamma_gassp:float
                               )->float:
    """
    :param p_MPaa: pressure in MPaa
    :param rsb_m3m3: gas-oil ratio at bubble poinr pressure, m3/m3
    :param t_K: temperature in K
    :param gamma_oil: specific oil density(by water)
    :param gamma_gassp: specific gas density(by air) in separator
    :return: reservoir free gas specific gravity

    ref1 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011
    """

    api = gamma_oil_2_api(gamma_oil)
    rsb_scfstb = m3m3_2_scfstb(rsb_m3m3)
    p_psia = MPa_2_psi(p_MPaa)
    t_F = K_2_F(t_K)
    gamma_gasr = 1 / (-208.0797 / p_psia + 22.885 / p_psia ** 2 - 0.000063641 * p_psia + 3.38346 / t_F ** 0.5 -
                      0.000992 * t_F - 0.000081147 * rsb_scfstb - 0.001956 * api + 1.081956 / gamma_gassp + 0.394035 *
                      gamma_gassp ** 2)
    return gamma_gasr

""" 
====================================================================================================
Расчет вязкости нефти
====================================================================================================
"""


def unf_viscosity_deadoil_Beggs_cP(t_K:float,
                                   gamma_oil:float, 
                                  )->float:
    """
        Correlation for dead oil viscosity

    :param gamma_oil: specific oil density (by water)
    :param t_K: temperature, K
    :return: dead oil viscosity,cP

    ref1 Beggs, H.D. and Robinson, J.R. “Estimating the Viscosity of Crude Oil Systems.”
    Journal of Petroleum Technology. Vol. 27, No. 9 (1975)

    """
    api = gamma_oil_2_api(gamma_oil)
    t_F = K_2_F(t_K)
    c = 10 ** (3.0324 - 0.02023 * api) * t_F ** -1.163  
    return 10 ** c - 1


def unf_viscosity_deadoil_BeggsRobinson_cP(t_K:float,
                                           gamma_oil:float, 
                                           )->float:
    """
    
    :param gamma_oil: specific oil density (by water)
    :param t_K: temperature, K
    :return: dead oil viscosity,cP

     формулы можно найти в PEH
        https://petrowiki.spe.org/PEH:Oil_System_Correlations
     точнее в приложении таблица A-7
        https://petrowiki.spe.org/File:Vol1_Page_323_Image_0001.png
    """
    # из vba  unf_pvt_viscosity_dead_oil_Beggs_Robinson_cP
    x = K_2_F(t_K) ** (-1.163) * np.exp(13.108 - 6.591 / gamma_oil)
    return 10 ** x - 1


def unf_viscosity_deadoil_Standing(t_K:float,
                                   gamma_oil:float, 
                                   )->float:
    """
    
    :param gamma_oil: specific oil density (by water)
    :param t_K: temperature, K
    :return: dead oil viscosity,cP

     похоже изначально это корреляция Beal 1946
       в оригинальное работе
       Beal, Carlton. "The Viscosity of Air, Water, Natural Gas, Crude Oil and Its Associated Gases at Oil Field Temperatures and Pressures." Trans. 165 (1946): 94–115. doi: https://doi.org/10.2118/946094-G
       нет формул только палетки
    
     изначально ссылка взята из  "Стандарт компании ЮКОС. Физические свойства нефти. Методы расчета" 2002
    
     формулы можно найти в PEH
        https://petrowiki.spe.org/PEH:Oil_System_Correlations
     точнее в приложении таблица A-7
        https://petrowiki.spe.org/File:Vol1_Page_323_Image_0001.png
    """

    # из vba  unf_pvt_viscosity_dead_oil_Standing_cP

    api = gamma_oil_2_api(gamma_oil)
    return (0.32 + 1.8e7 / api ** 4.53) * (360 / (K_2_F(t_K) + 200)) ** (10 ** (0.43 + 8.33 / api))


def unf_viscosity_oil_Standing_cP(p_MPa:float,  
                                  rs_m3m3:float,
                                  pb_MPa:float, 
                                  mu_oil_dead_cP:float
                                  )->float:

    a = 5.6148 * rs_m3m3 * (0.1235 * 10 ** (-5) * rs_m3m3 - 0.00074)
    b = 0.68 / 10 ** (0.000484 * rs_m3m3) + 0.25 / 10 ** (0.006176 * rs_m3m3) + 0.062 / 10 ** (0.021 * rs_m3m3)

    unf_pvt_viscosity_oil_Standing_cP = 10 ** a * mu_oil_dead_cP ** b

    return np.where(pb_MPa < p_MPa,
                    unf_pvt_viscosity_oil_Standing_cP + 0.14504 * (p_MPa - pb_MPa) * (0.024 * unf_pvt_viscosity_oil_Standing_cP ** 1.6 + 0.038 * unf_pvt_viscosity_oil_Standing_cP ** 0.56),
                    unf_pvt_viscosity_oil_Standing_cP
                    )


def unf_viscosity_saturatedoil_Beggs_cP(mu_oil_dead_cP:float, 
                                       rs_m3m3:float
                                       )->float:
    """
        Correlation for oil viscosity for pressure below bubble point (for pb!!!)

    :param mu_oil_dead_cP: dead oil viscosity,cP
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :return: oil viscosity,cP

    ref1 Beggs, H.D. and Robinson, J.R. “Estimating the Viscosity of Crude Oil Systems.”
    Journal of Petroleum Technology. Vol. 27, No. 9 (1975)

    """
    rs_scfstb = m3m3_2_scfstb(rs_m3m3)
    a = 10.715 * (rs_scfstb + 100) ** -0.515
    b = 5.44 * (rs_scfstb + 150) ** -0.338
    return a * mu_oil_dead_cP ** b


def unf_viscosity_undersaturatedoil_VB_cP(p_MPaa:float, 
                                          pb_MPaa:float, 
                                          mu_oilb_cP:float
                                          )->float:
    """
        Viscosity correlation for pressure above bubble point

    :param p_MPaa: pressure, MPaa
    :param pb_MPaa: bubble point pressure, MPaa
    :param mu_oilb_cP: oil viscosity at bubble point pressure, cP
    :return: oil viscosity,cP

    ref2 Vazquez, M. and Beggs, H.D. 1980. Correlations for Fluid Physical Property Prediction.
    J Pet Technol 32 (6): 968-970. SPE-6719-PA

    """
    p_psia = MPa_2_psi(p_MPaa)
    pb_psia = MPa_2_psi(pb_MPaa)
    m = 2.6 * p_psia ** 1.187 * np.exp(-11.513 - 8.98e-5 * p_psia)
    viscosity_cP = mu_oilb_cP * (p_psia / pb_psia) ** m
    return viscosity_cP


def unf_viscosity_oil_Beggs_VB_cP(p_MPaa:float, 
                                  rs_m3m3:float, 
                                  pb_MPaa:float,
                                  mu_oil_dead_cP:float, 
                                  )->float:
    """
        Function for calculating the viscosity at any pressure

    :param mu_oil_dead_cP: dead oil viscosity,cP
    :param rs_m3m3: solution gas-oil ratio, m3m3
    :param p_MPaa: pressure, MPaa
    :param pb_MPaa: bubble point pressure, MPaa
    :return: oil viscosity,cP
    """
    
    saturatedviscosity_cP = unf_viscosity_saturatedoil_Beggs_cP(mu_oil_dead_cP, rs_m3m3)
    return np.where(p_MPaa <= pb_MPaa, 
                    saturatedviscosity_cP,
                    unf_viscosity_undersaturatedoil_VB_cP(p_MPaa, pb_MPaa, saturatedviscosity_cP)
                    )


def unf_viscosity_undersaturatedoil_Petrosky_cP(p_MPaa:float, 
                                                pb_MPaa:float, 
                                                mu_oilb_cP:float
                                                )->float:
    """
        Viscosity correlation for pressure above bubble point

    :param p_MPaa: pressure, MPaa
    :param pb_MPaa: bubble point pressure, MPaa
    :param mu_oilb_cP: oil viscosity at bubble point pressure, cP
    :return: oil viscosity,cP

    ref 1 Petrosky, G.E. and Farshad, F.F. “Viscosity Correlations for Gulf of Mexico Crude
    Oils.” Paper SPE 29468. Presented at the SPE Production Operations Symposium,
    Oklahoma City (1995)
    """
    log_mu_oilb_cP = np.log(mu_oilb_cP)
    A = -1.0146 + 1.3322 * log_mu_oilb_cP - 0.4876 * log_mu_oilb_cP ** 2 - 1.15036 * log_mu_oilb_cP ** 3
    return  mu_oilb_cP + 1.3449e-3 * MPa_2_psi(p_MPaa-pb_MPaa) * 10 ** A


""" 
====================================================================================================
Расчет тепловых свойств
====================================================================================================
"""

def unf_heat_capacity_oil_Gambill_JkgC(gamma_oil:float, 
                                       t_C:float
                                       )->float:
    """
        Oil heat capacity in SI. Gambill correlation

    :param gamma_oil: specific oil density(by water)
    :param t_c: temperature in C
    :return: heat capacity in SI - JkgC

    ref1 Book: Brill J. P., Mukherjee H. K. Multiphase flow in wells. –
    Society of Petroleum Engineers, 1999. – Т. 17. in Page 122
    """

    t_F = C_2_F(t_C)
    heat_capacity_oil_btulbmF = ((0.388 + 0.00045 * t_F) / gamma_oil ** (0.5))
    return btulbmF_2_kJkgK(heat_capacity_oil_btulbmF) * 1000


def unf_heat_capacity_oil_Wes_Wright_JkgC(gamma_oil:float, 
                                          t_C:float
                                          )->float:
    """
        Oil heat capacity in SI. Wes Wright method

    :param gamma_oil: specific oil density(by water)
    :param t_c: temperature in C
    :return: heat capacity in SI - JkgC

    ref1 https://www.petroskills.com/blog/entry/crude-oil-and-changing-temperature#.XQkEnogzaM8
    """
    return ((2e-3 * t_C - 1.429 ) * gamma_oil + (2.67e-3) * t_C + 3.049) * 1000


def unf_thermal_conductivity_oil_Abdul_Seoud_Moharam_WmK(gamma_oil:float, 
                                                         t_C:float
                                                         )->float:
    """
        Oil thermal conductivity Abdul-Seoud and Moharam correlation

    :param gamma_oil: specific oil density(by water)
    :param t_c: temperature in C
    :return: thermal conductivity in SI - wt / m K

    ref1 Tovar L. P. et al. Overview and computational approach for studying the physicochemical characterization
    of high-boiling-point petroleum fractions (350 C+) //
    Oil & Gas Science and Technology–Revue d’IFP Energies nouvelles. – 2012. – Т. 67. – №. 3. – С. 451-477.
    """
 
    return (2.540312 * (gamma_oil / C_2_K(t_C)) ** 0.5) - 0.014485


def unf_thermal_conductivity_oil_Smith_WmK(gamma_oil:float, 
                                           t_C:float
                                           )->float:
    """
        Oil thermal conductivity Smith correlation for 273 < T < 423 K

    :param gamma_oil: specific oil density(by water)
    :param t_c: temperature in C
    :return: thermal conductivity in SI - wt / m K

    ref1 Das D. K., Nerella S., Kulkarni D. Thermal properties of petroleum and gas-to-liquid products //
    Petroleum science and technology. – 2007. – Т. 25. – №. 4. – С. 415-425.   """

    return (0.137 / (gamma_oil * 1000) * (1 - 0.00054 * (C_2_K(t_C) - 273)) * 1e3)


def unf_thermal_conductivity_oil_Cragoe_WmK(gamma_oil:float, 
                                            t_C:float
                                            )->float:
    """
        Oil thermal conductivity Cragoe correlation for 273 < T < 423 K

    :param gamma_oil: specific oil density(by water)
    :param t_c: temperature in C
    :return: thermal conductivity in SI - wt / m K

    ref1 Das D. K., Nerella S., Kulkarni D. Thermal properties of petroleum and gas-to-liquid products //
    Petroleum science and technology. – 2007. – Т. 25. – №. 4. – С. 415-425.   """
    t_K = C_2_K(t_C)
    return (0.118 /(gamma_oil * 1000) * (1 - 0.00054 * (t_K - 273)) * 10 ** 3)



    # uPVT свойства для сжимаемости нефти(требует немного свойств газа)

"""
def unf_weightedcompressibility_oil_Mccain_1MPa_greater(gamma_oil, gamma_gas, pb_MPa, p_MPa, rsb_m3m3, tres_K, gamma_gassp = 0):

    pass


def unf_compressibility_oil_Mccain_1MPa_greater(gamma_oil, gamma_gas, pb_MPa, p_MPa, rsb_m3m3, tres_K, gamma_gassp = 0):

    pass


def unf_compressibility_oil_Mccain_1MPa_lower():
    pass
"""