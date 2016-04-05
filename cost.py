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
        