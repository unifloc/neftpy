import neftpy.upvt_gas as pvtg

import numpy as np
import scipy.optimize as opt

ppr = 1
tpr = 1
gg = 0.8


def unf_zfactor_DAK_ppr(ppr:float, tpr:float)->float:
    """
        Correlation for z-factor

    :param ppr: pseudoreduced pressure
    :param tpr: pseudoreduced temperature
    :return: z-factor

    range of applicability is (0.2<=ppr<30 and 1.0<tpr<=3.0) and also ppr < 1.0 for 0.7 < tpr < 1.0

    ref 1 Dranchuk, P.M. and Abou-Kassem, J.H. “Calculation of Z Factors for Natural
    Gases Using Equations of State.” Journal of Canadian Petroleum Technology. (July–September 1975) 34–36.

    """

    z0 = 1
    ropr0 = 0.27 * (ppr / (z0 * tpr))

    def f(variables):
        z, ropr = variables
        func = np.zeros(2)
        func[0] = 0.27 * (ppr / (z * tpr)) - ropr
        func[1] = -z + 1 + \
                (0.3265 - 1.0700 / tpr - 0.5339 / tpr**3 + 0.01569 / tpr ** 4 - 0.05165 / tpr ** 5) * ropr +\
                (0.5475 - 0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 2 - \
                0.1056 * (-0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 5 + \
                0.6134 * (1 + 0.7210 * ropr ** 2) * (ropr ** 2 / tpr ** 3) * np.exp(-0.7210 * ropr ** 2)
        print(variables, func)
        return func
    solution = opt.fsolve(f, np.array([z0, ropr0]))
    """
    solution = opt.newton(f, z0, maxiter=150, tol=1e-4)
    """
    return solution[0]




def unf_zfactor_DAK_ppr_(ppr:float, tpr:float)->float:
    """
        Correlation for z-factor

    :param ppr: pseudoreduced pressure
    :param tpr: pseudoreduced temperature
    :return: z-factor

    range of applicability is (0.2<=ppr<30 and 1.0<tpr<=3.0) and also ppr < 1.0 for 0.7 < tpr < 1.0

    ref 1 Dranchuk, P.M. and Abou-Kassem, J.H. “Calculation of Z Factors for Natural
    Gases Using Equations of State.” Journal of Canadian Petroleum Technology. (July–September 1975) 34–36.

    """

    z0 = 1
    ropr0 = 0.27 * (ppr / (z0 * tpr))

    def f(z):
        #z, ropr = variables
        #func = np.zeros(2)
        ropr = 0.27 * (ppr / (z * tpr)) 
        func = -z + 1 + \
                (0.3265 - 1.0700 / tpr - 0.5339 / tpr**3 + 0.01569 / tpr ** 4 - 0.05165 / tpr ** 5) * ropr +\
                (0.5475 - 0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 2 - \
                0.1056 * (-0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 5 + \
                0.6134 * (1 + 0.7210 * ropr ** 2) * (ropr ** 2 / tpr ** 3) * np.exp(-0.7210 * ropr ** 2)
        print(z, func)
        return func
    solution = opt.newton(f, z0, maxiter=150, tol=1e-5)
    """
    solution = opt.fsolve(f, np.array([z0]))
    """
    return solution

print(pvtg.unf_pseudocritical_temperature_Standing_K(gg))

z = unf_zfactor_DAK_ppr(ppr, tpr)
print(z)


z1 = unf_zfactor_DAK_ppr_(ppr, tpr)
print(z1)