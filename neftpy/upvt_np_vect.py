# векторизованные функции расчета pvt свойств 
from neftpy.uconvert import *
from neftpy.uconst import *

import neftpy.upvt_oil as pvto

import numpy as np



# наивная векторизация расчета давления насыщения по Стендингу
unf_pb_Standing_MPaa = np.vectorize(pvto.unf_pb_Standing_MPaa)

# наивная векторизация расчета давления насыщения по Валко Мак Кейну
unf_pb_Valko_MPaa = np.vectorize(pvto.unf_pb_Valko_MPaa)

# газосодержание по Стендингу
unf_rs_Standing_m3m3 = np.vectorize(pvto.unf_rs_Standing_m3m3)

# наивная векторизация
unf_rs_Velarde_m3m3 = np.vectorize(pvto.unf_rs_Velarde_m3m3)

# наивная векторизация
unf_rsb_Mccain_m3m3 = np.vectorize(pvto.unf_rsb_Mccain_m3m3)