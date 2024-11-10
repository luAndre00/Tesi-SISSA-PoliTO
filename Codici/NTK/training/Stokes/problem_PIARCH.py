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
from pina.operators import laplacian, grad

alpha = 0.008

class ParametricStokesOptimalControl(SpatialProblem, ParametricProblem):
    # assign output/ spatial variables
    output_variables = ['vx', 'vy', 'p', 'ux', 'uy', 'r', 'zx', 'zy']
    # vx, vy is the 2-dim state variable, #ux, uy is the 2-dim control variable.
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 2
    mumin = 0.5
    mumax = 1.5

    xrange = [xmin, xmax]
    yrange = [ymin, ymax]
    murange = [mumin, mumax]

    # r è la variabile aggiunta della pressione, z è la aggiunta del campo di velocità
    spatial_domain = CartesianDomain({'x': xrange, 'y': yrange})
    parameter_domain = CartesianDomain({'mu': murange})

    ##############################################################################

    # Prima ci sono tutte le equazioni sulle variabili aggiunte, poi tutte quelle per le variabili non aggiunte
    # PDE
    def momentum_ad_x(input_, output_):
        delta = laplacian(output_, input_, components=['ux'], d=['x', 'y'])
        return -0.1 * alpha * delta + grad(output_, input_, components=['r'], d=['x']) - input_.extract(['y']) + output_.extract(['vx'])

    def momentum_ad_y(input_, output_):
        delta = laplacian(output_, input_, components=['uy'], d=['x', 'y'])
        return -0.1 * alpha * delta + grad(output_, input_, components=['r'], d=['y'])

    def continuity_ad(input_, output_):
        return grad(output_, input_, components=['ux'], d=['x']) + grad(output_, input_, components=['uy'], d=['y'])

    # BOUNDARY CONDITIONS on adjuncted variables
    # Dirichlet
    def dirichlet1_ad(input_, output_): 
        return output_.extract(['ux'])

    def dirichlet2_ad(input_, output_):
        return output_.extract(['uy'])

    # Neumann
    def neumann1_ad(input_, output_):
        return -output_.extract(['r']) + 0.1 * alpha * grad(output_, input_, components=['ux'], d=['x'])

    def neumann2_ad(input_, output_):
        return output_.extract(['uy'])

    ############################################################################

    # Momentum Equations
    def momentum_x(input_, output_):
        delta = laplacian(output_, input_, components=['vx'], d=['x', 'y'])
        return -0.1 * delta + grad(output_, input_, components=['p'], d=['x']) - output_.extract(['ux'])

    def momentum_y(input_, output_):
        delta = laplacian(output_, input_, components=['vy'], d=['x', 'y'])
        return -0.1 * delta + grad(output_, input_, components=['p'], d=['y']) - input_.extract(
            ['mu']) - output_.extract(['uy'])

    def continuity(input_, output_):
        return grad(output_, input_, components=['vx'], d=['x']) + grad(output_, input_, components=['vy'], d=['y'])

    # BOUNDARY CONDITIONS on principal variable
    # Dirichlet
    def dirichlet1(input_, output_):
        return output_.extract(['vx']) - input_.extract(['y'])

    def dirichlet2(input_, output_):
        return output_.extract(['vy'])

    # Neumann
    def neumann1(input_, output_):
        return -output_.extract(['p']) + 0.1 * grad(output_, input_, components=['vx'], d=['x'])

    def neumann2(input_, output_):
        return output_.extract(['vy'])

    # Problem Statement
    conditions = {
        #'gamma_above': Condition(location=CartesianDomain({'x': xrange, 'y': 2, 'mu': murange}), equation=SystemEquation([dirichlet1, dirichlet2, dirichlet1_ad, dirichlet2_ad], reduction = "none")),
        # Dirichlet
        #'gamma_left': Condition(location=CartesianDomain({'x': 0, 'y': yrange, 'mu': murange}), equation=SystemEquation([dirichlet1, dirichlet2, dirichlet1_ad, dirichlet2_ad], reduction = "none")),
        # Dirichlet
        #'gamma_below': Condition(location=CartesianDomain({'x': xrange, 'y': 0, 'mu': murange}), equation=SystemEquation([dirichlet1, dirichlet2, dirichlet1_ad, dirichlet2_ad], reduction = "none")),
        # Dirichlet
        'state': Condition(location=CartesianDomain({'x': xrange, 'y': yrange, 'mu': murange}), equation=SystemEquation([momentum_x, momentum_y, continuity], reduction = "none")),
        'adjoint': Condition(location=CartesianDomain({'x': xrange, 'y': yrange, 'mu': murange}), equation=SystemEquation([momentum_ad_x, momentum_ad_y, continuity_ad], reduction = "none")),
        'gamma_right': Condition(location=CartesianDomain({'x': 1, 'y': yrange, 'mu': murange}), equation=SystemEquation([neumann1, neumann2, neumann1_ad, neumann2_ad], reduction="none"))}
