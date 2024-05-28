""" Poisson OCP problem. """


from pina import Condition
from pina.geometry import CartesianDomain
from pina.equation import SystemEquation, FixedValue
from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import laplacian

# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Parametric Elliptic Optimal Control problem.         #
#  The ParametricEllipticOptimalControl class is        #
#  inherited from TimeDependentProblem, SpatialProblem  #
#  and we denote:                                       #
#           u --> field variable                        #
#           p --> field variable                        #
#           y --> field variable                        #
#           x1, x2 --> spatial variables                #
#           mu, alpha --> problem parameters            #
#                                                       #
#  More info in https://arxiv.org/pdf/2110.13530.pdf    #
#  Section 4.2 of the article                           #
# ===================================================== #


class ParametricEllipticOptimalControl(SpatialProblem, ParametricProblem):

    # setting spatial variables ranges
    xmin, xmax, ymin, ymax = -1, 1, -1, 1
    x_range = [xmin, xmax]
    y_range = [ymin, ymax]
    # setting parameters range
    a = 1
    mumin, mumax = 0.5, 3
    mu_range = [mumin, mumax]
    # setting field variables
    output_variables = ['u', 'p', 'y']
    # setting spatial and parameter domain
    spatial_domain = CartesianDomain({'x1': x_range, 'x2': y_range})
    parameter_domain = CartesianDomain({'mu': mu_range})

    #Questa è la prima equazione della equazione 4.9
    def term1(input_, output_):
        laplace_p = laplacian(output_, input_, components=['p'], d=['x1', 'x2'])
        return output_.extract(['y']) - input_.extract(['mu']) - laplace_p

    #Questa è la quarta equazione della equazione 4.9
    def term2(input_, output_):
        laplace_y = laplacian(output_, input_, components=['y'], d=['x1', 'x2'])
        return - laplace_y - output_.extract(['u'])

    #Questa è la quinta equazione della equazione 4.9
    def fixed_y(input_, output_):
        return output_.extract(['y'])

    # #Questa è la seconda equazione della equazione 4.9
    def fixed_p(input_, output_):
        return output_.extract(['p'])

    # setting problem condition formulation
    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x1': x_range, 'x2':  1, 'mu': mu_range}),
            equation=SystemEquation([fixed_y, fixed_p])),
        'gamma2': Condition(
            location=CartesianDomain({'x1': x_range, 'x2': -1, 'mu': mu_range}),
            equation=SystemEquation([fixed_y, fixed_p])),
        'gamma3': Condition(
            location=CartesianDomain({'x1':  1, 'x2': y_range, 'mu': mu_range}),
            equation=SystemEquation([fixed_y, fixed_p])),
        'gamma4': Condition(
            location=CartesianDomain({'x1': -1, 'x2': y_range, 'mu': mu_range}),
            equation=SystemEquation([fixed_y, fixed_p])),
        'D': Condition(
            location=CartesianDomain(
                {'x1': x_range, 'x2': y_range,
                'mu': mu_range
                }),
            equation=SystemEquation([term1, term2])), #Ho aggiunto qui term3
    }
