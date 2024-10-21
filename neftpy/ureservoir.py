"""
расчет характеристик пласта, начиная с корреляции Вогеля
"""

from neftpy.uconvert import *
from neftpy.uconst import *

import numpy as np

def _ipr_Vogel_qliq_sm3day(p_test_atma:float, 
                          p_res_atma:float, 
                          pi_sm3dayatm:float, 
                          fw_perc:float,
                          pb_atma:float=0
                          )->float:
    """
    Calculate well liquid rate using Vogel's method with water cut correction.

    Args:
        p_test_atma (float): Test pressure, atm
        p_res_atma (float): Reservoir pressure, atm
        pi_sm3dayatm (float): Productivity index, m3/day/atm
        fw_perc (float): Water cut, %
        pb_atma (float): Bubble point pressure, atm

    Returns:
        float: Well bottom pressure, atm
    """
    if p_test_atma < 0:
        raise ValueError("P_test cannot be negative")
    if p_res_atma < 0:
        raise ValueError("Pr cannot be negative")
    if pb_atma < 0:
        raise ValueError("Pb cannot be negative")
    if pi_sm3dayatm < 0:
        raise ValueError("pi cannot be negative")

    if p_res_atma < pb_atma:
        pb_atma = p_res_atma

    qb = pi_sm3dayatm * (p_res_atma - pb_atma)

    if fw_perc > 100:
        fw_perc = 100
    if fw_perc < 0:
        fw_perc = 0

    if fw_perc == 100 or p_test_atma >= pb_atma:
        return pi_sm3dayatm * (p_res_atma - p_test_atma)
    else:
        fw = fw_perc / 100
        fo = 1 - fw
        qo_max = qb + (pi_sm3dayatm * pb_atma) / 1.8
        p_wfg = fw * (p_res_atma - qo_max / pi_sm3dayatm)

        if p_test_atma > p_wfg:
            a = 1 + (p_test_atma - fw * p_res_atma) / (0.125 * fo * pb_atma)
            b = fw / (0.125 * fo * pb_atma * pi_sm3dayatm)
            c = (2 * a * b) + 80 / (qo_max - qb)
            d = (a ** 2) - (80 * qb / (qo_max - qb)) - 81
            if b == 0:
                return abs(d / c)
            else:
                return (-c + ((c ** 2 - 4 * b ** 2 * d) ** 0.5)) / (2 * b ** 2)
        else:
            cg = 0.001 * qo_max
            cd = fw * (cg / pi_sm3dayatm) + fo * 0.125 * pb_atma * (-1 + (1 + 80 * ((0.001 * qo_max) / (qo_max - qb))) ** 0.5)
            return (p_wfg - p_test_atma) / (cd / cg) + qo_max

ipr_Vogel_qliq_sm3day = np.vectorize(_ipr_Vogel_qliq_sm3day)

# ==============================================================

def _ipr_Vogel_pwf_atma(q_test_sm3day:float, 
                       p_res_atma:float, 
                       pi_sm3dayatm:float, 
                       fw_perc:float, 
                       pb_atma:float=0
                       )->float:
    """
    Calculate well bottom pressure using Vogel's method with water cut correction.

    Args:
        q_test_sm3day (float): Test flow rate, m3/day
        p_res_atma (float): Reservoir pressure, atm
        pi_sm3dayatm (float): Productivity index, m3/day/atm
        fw_perc (float): Water cut, %
        pb_atma (float): Bubble point pressure, atm

    Returns:
        float: Well bottom pressure, atm
    """
    if p_res_atma < pb_atma:
        pb_atma = p_res_atma

    if q_test_sm3day < 0:
        raise ValueError("Q_test cannot be negative")
    if p_res_atma <= 0:
        raise ValueError("Pr cannot be negative or zero")
    if pb_atma < 0:
        raise ValueError("Pb cannot be negative")
    if pi_sm3dayatm <= 0:
        raise ValueError("pi cannot be negative or zero")

    qb = pi_sm3dayatm * (p_res_atma - pb_atma)
    if fw_perc > 100:
        fw_perc = 100
    if fw_perc < 0:
        fw_perc = 0

    if fw_perc == 100 or q_test_sm3day <= qb or pb_atma == 0:
        return p_res_atma - q_test_sm3day / pi_sm3dayatm
    else:
        fw = fw_perc / 100
        fo = 1 - fw
        qo_max = qb + (pi_sm3dayatm * pb_atma) / 1.8

        if q_test_sm3day < qo_max:
            return fw * (p_res_atma - q_test_sm3day / pi_sm3dayatm) + fo * 0.125 * pb_atma * (-1 + (1 - 80 * ((q_test_sm3day - qo_max) / (qo_max - qb))) ** 0.5)
        else:
            cg = 0.001 * qo_max
            cd = fw * (cg / pi_sm3dayatm) + fo * 0.125 * pb_atma * (-1 + (1 + 80 * ((0.001 * qo_max) / (qo_max - qb))) ** 0.5)
            return fw * (p_res_atma - qo_max / pi_sm3dayatm) - (q_test_sm3day - qo_max) * (cd / cg)

    if ipr_Vogel_pwf_atma < 0:
        return 0

ipr_Vogel_pwf_atma = np.vectorize(_ipr_Vogel_pwf_atma)

# ===============================================

def _ipr_Vogel_pi_sm3dayatm(q_test_sm3day:float, 
                            p_test_sm3day:float, 
                            p_res_sm3day:float, 
                            fw_perc:float, 
                            pb_atma:float=0)->float:
    """
    Calculate productivity index using Vogel's method with water cut correction.

    Args:
        q_test_sm3day (float): Test flow rate, m3/day
        p_test_sm3day (float): Test pressure, atm
        p_res_sm3day (float): Reservoir pressure, atm
        pb_atma (float): Bubble point pressure, atm
        fw_perc (float): Water cut, %

    Returns:
        float: Productivity index, m3/day/atm
    """
    if p_test_sm3day < 0:
        p_test_sm3day = 0
        return 0

    if p_res_sm3day < pb_atma:
        pb_atma = p_res_sm3day

    if q_test_sm3day <= 0:
        return 0
    if p_test_sm3day <= 0:
        return 0
    if pb_atma < 0:
        return 0
    if p_res_sm3day <= 0:
        return 0

    j = q_test_sm3day / (p_res_sm3day - p_test_sm3day)
    Q_calibr = _ipr_Vogel_qliq_sm3day(p_test_sm3day, p_res_sm3day, j, fw_perc, pb_atma)
    j = j / ((Q_calibr) / q_test_sm3day)
    Q_calibr = _ipr_Vogel_qliq_sm3day(p_test_sm3day, p_res_sm3day, j, fw_perc, pb_atma)
    if abs(Q_calibr - q_test_sm3day) > 0.001:
        raise ValueError("Calculation failed")

    return j

ipr_Vogel_pi_sm3dayatm = np.vectorize(_ipr_Vogel_pi_sm3dayatm)