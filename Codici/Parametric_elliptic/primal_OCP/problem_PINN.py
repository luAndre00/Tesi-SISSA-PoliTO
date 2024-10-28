# import sys
# sys.path.append('C:/Users/Andrea/Desktop/Poli/Tesi magistrale/reporitory_SISSA_PoliTO')

#Problem definition for PINN

import numpy as np
import torch
from pina import LabelTensor
from pina.model import FeedForward

from pina import Condition
from pina.geometry import CartesianDomain
from pina.equation import SystemEquation, FixedValue
from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import laplacian

class ParametricEllipticOptimalControl(SpatialProblem, ParametricProblem):
    xmin, xmax, ymin, ymax = -1, 1, -1, 1    # setting spatial variables ranges
    x_range = [xmin, xmax]
    y_range = [ymin, ymax]
    # setting parameters range
    mu1min, mu1max = 0.5, 3
    mu2min, mu2max = 0.01, 1
    mu1_range = [mu1min, mu1max]
    mu2_range = [mu2min, mu2max]
    output_variables = ['u', 'y', 'z']    # setting field variables
    # setting spatial and parameter domain
    spatial_domain = CartesianDomain({'x1': x_range, 'x2': y_range})
    parameter_domain = CartesianDomain({'mu1': mu1_range, 'mu2': mu2_range})

    def laplace(input_, output_):
        laplace_y = laplacian(output_, input_, components=['y'], d=['x1', 'x2'])
        return - laplace_y - output_.extract(['u'])

    # setting problem condition formulation
    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x1': x_range, 'x2': 1, 'mu1': mu1_range, 'mu2': mu2_range}),
            equation=FixedValue(0, ['z', 'y'])),
        'gamma2': Condition(
            location=CartesianDomain({'x1': x_range, 'x2': -1, 'mu1': mu1_range, 'mu2': mu2_range}),
            equation=FixedValue(0, ['z', 'y'])),
        'gamma3': Condition(
            location=CartesianDomain({'x1': 1, 'x2': y_range, 'mu1': mu1_range, 'mu2': mu2_range}),
            equation=FixedValue(0, ['z', 'y'])),
        'gamma4': Condition(
            location=CartesianDomain({'x1': -1, 'x2': y_range, 'mu1': mu1_range, 'mu2': mu2_range}),
            equation=FixedValue(0, ['z', 'y'])),
        'D': Condition(location=CartesianDomain({'x1': x_range, 'x2': y_range, 'mu1': mu1_range, 'mu2': mu2_range}),
                       equation=SystemEquation([laplace], reduction='none')), #Senza il reduction none c'è un bias nella loss
    }