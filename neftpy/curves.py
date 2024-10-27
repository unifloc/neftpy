"""
вспомогательные классы для работы с кривыми - параметрами дволь одной оси
"""

import numpy as np
import scipy.interpolate as scipy_interp
from numpy.typing import ArrayLike



class Curve:
    """
    класс описывающий кривую f(x) - таблично заданную функцию
    """

    def __init__(self):
        
        self._empty = True

    def add_point(self, x:float, y:float):
        """
        добавляем в массив точки
        в итоге значения x должны быть уникальны
        """

        if self._empty:
            self._points = np.array([x,y]).T

        # удалим из массива точки которые хотим добавить если были
        pnts = self._points[np.isin(self._points[:,0], x, invert=True)]
        # добавим снизу
        pnts = a = np.vstack([pnts, np.array([x,y]).T])  

        # отсортируем все по x       
        self._points = pnts[pnts[:,0].argsort()] 
        self._make_linear_interpolant()               

        self._empty = False

    def _make_linear_interpolant(self):
        # сформируем объект для интерполяции
        bspl = scipy_interp.make_interp_spline(x=self._points[:,0],
                                               y=self._points[:,1],
                                               k=1)
        self._linear_interpolant = scipy_interp.PPoly.from_spline(bspl)  

    @property
    def points(self):
        return self._points
    
    @property
    def points_x(self):
        return self._points[:,0]
    
    @property
    def points_y(self):
        return self._points[:,1]
    
    @property
    def num_points(self):
        return len(self._points)
    
    def _interp(self, xval:ArrayLike):
        """
        интерполяция значений с использованием numpy interp
        """
        return np.interp(x=xval, 
                         xp=self._points[:,0],
                         fp=self._points[:,1])


