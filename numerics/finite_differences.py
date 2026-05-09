#Implemented methods:
#1. Forward, backward, central differences
#2. Richardson extrapolation for higher accuracy

import numpy as np

def forward_difference(f, x, h):
    return (f(x+h) - f(x))/h

def backward_difference(f,x,h):
    return (f(x) - f(x-h))/h

def central_difference(f, x, h):
    return (f(x+h) - f(x-h))/(2*h)

def second_derivative(f, x, h):
    return (f(x+h) - 2*f(x) + f(x-h))/ h**2

def richardson_extrapolation(f, x, h, order=2):
    D_h = central_difference(f,x,h)
    D_half_h = central_difference(f,x, h/2)
    factor = 2 ** order
    return (factor * D_half_h - D_h) / (factor - 1)

def differentiate(y, dt):
    n = len(y)
    dy = np.zeros(n)
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    dy[0]  = (y[1]  - y[0])/dt
    dy[-1] = (y[-1] - y[-2])/dt
    return dy

def differentiate_richardson(y, dt):
    n = len(y)
    dy = np.zeros(n)
    dy[2:-2] = (-y[4:] + 8 * y[3:-1] - 8 * y[1:-3] + y[:-4]) / (12 * dt)
    dy[1] = (y[2] - y[0]) / (2 * dt)
    dy[-2] = (y[-1] - y[-3]) / (2 * dt)
    dy[0] = (y[1] - y[0]) / dt
    dy[-1] = (y[-1] - y[-2]) / dt
    return dy