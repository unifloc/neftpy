import neftpy.curves as crv
import numpy as np

c = crv.Curve()
x = np.linspace(1, 10, 2)
y = x**2

c.add_point(x,y)
c.add_point(2, 34)
print(c.points)
print(c._interp([-1.3,3.3, 5.5]))
print(c._linear_interpolant([-1.3,3.3, 5.5]))