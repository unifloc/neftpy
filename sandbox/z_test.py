import neftpy.upvt_gas as pvtg

import numpy as np
import scipy.optimize as opt

ppr = 2
tpr = 1.2
gg = 0.8



print(pvtg.unf_pseudocritical_temperature_Standing_K(gg))

z = pvtg.unf_zfactor_DAK_ppr(ppr, tpr)
print(z)


z1 = pvtg.unf_zfactor_SK(ppr, tpr)
print(z1)