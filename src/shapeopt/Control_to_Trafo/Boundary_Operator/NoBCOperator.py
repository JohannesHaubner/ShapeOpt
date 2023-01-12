from dolfin import *
from .BoundaryOperator import BoundaryOperator
import numpy as np

class NoBCO(BoundaryOperator):
    def __init__(self, dmesh, dnormal, lb_off):
        super().__init__(dmesh, dnormal, lb_off)

    def eval(self, x):
        # x: corresponds to control in self.Vd
        return x

    def chainrule(self, djy):
        # compute derivative of j(vector_laplace_beltrami(x)) under the knowledge of
        # djy = nabla j(y) (gradient)
        return djy.vector()