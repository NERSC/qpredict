from neon import NervanaObject
import numpy as np

class Cost(NervanaObject):

    """
    Base class for the cost functions
    """

    def __call__(self, y, t):
        """
        Applies the cost function
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y
        Returns:
            OpTree: Returns the cost
        """
        return self.func(y, t)

    def bprop(self, y, t):
        """
        Computes the derivative of the cost function
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y
        Returns:
            OpTree: Returns the derivative of the cost function
        """
        return self.funcgrad(y, t)


class MeanSquared(Cost):

    def __init__(self):
        """
        Initialize the squared error cost functions
        """
        
        self.func = lambda y, t: self.be.mean(
            self.be.square(y - t), axis=0) / 2.
        self.funcgrad = lambda y, t: (y - t)


class SmoothL1Loss(Cost):

    """
    A smooth L1 loss cost function from Fast-RCNN
    http://arxiv.org/pdf/1504.08083v2.pdf
    L1 loss is less sensitive to the outlier than L2 loss used in RCNN
    """

    def smoothL1(self, x):
        return (0.5 * self.be.square(x) * (self.be.absolute(x) < 1) +
                (self.be.absolute(x) - 0.5) * (self.be.absolute(x) >= 1))

    def smoothL1grad(self, x):
        return (x * (self.be.absolute(x) < 1) + self.be.sgn(x) *
                (self.be.absolute(x) >= 1))

    def __init__(self):
        """
        Initialize the smooth L1 loss cost function
        """
        self.func = lambda y, t: self.be.sum(self.smoothL1(y - t), axis=0)
        self.funcgrad = lambda y, t: self.smoothL1grad(y - t)
        