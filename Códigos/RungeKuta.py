# RungeKuta.py

# Solucion de ecuaciones diferenciales por Runge Kuta

import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4th_order(func, y0, t_span, h):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = []

    y_current = np.array(y0)
    for t in t_values:
        y_values.append(y_current)
        k1 = h * func(t, y_current)
        k2 = h * func(t + 0.5 * h, y_current + 0.5 * k1)
        k3 = h * func(t + 0.5 * h, y_current + 0.5 * k2)
        k4 = h * func(t + h, y_current + k3)
        y_current = y_current + (k1 + 2*k2 + 2*k3 + k4) / 6

    return np.array(t_values), np.array(y_values)

def CrearFuncionLorenz(a,b,c)
    def AtrayenteLorenz(t, y):
        dy1dt = a*(y[1]-y[0])
        dy2dt = y[0]*(b-y[2])-y[1]
        dy3dt = y[0]*y[1]-c*y[2]
        return np.array([dy1dt, dy2dt, dy3dt])
    return AtrayenteLorenz

def CrearFuncionRossler(a,b,c):
    def AtrayenteRossler(t, y):
        dy1dt = -y[1]-y[2]
        dy2dt = y[0]+a*y[1]
        dy3dt = b+y[2]*(y[1]-c)
        return np.array([dy1dt, dy2dt, dy3dt])
    return AtrayenteRossler