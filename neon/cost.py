from neon import NervanaObject
import numpy as np

#general cost functions
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


#specialized cost functions
class MeanSquaredLoss(Cost):

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
        self.func = lambda y, t: self.be.mean(self.smoothL1(y - t), axis=0)
        self.funcgrad = lambda y, t: self.smoothL1grad(y - t)


#general metric class
class Metric(Cost):

    """
    Base class for Metric
    Meant for non-smooth costs that we just want to check on validation.
    """

    def __call__(self, y, t):
        """
        To implement in derived classes
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y
        Returns:
            float: Returns the metric
        """
        raise NotImplementedError()

    def bprop(self, y, t):
        """
        Not relevant for Metric
        """
        pass


#specialized metrics:
class MeanSquaredMetric(Metric):

    """
    Compute the MSQMetric
    """

    def __init__(self):
        self.outputs = self.be.iobuf(1)  # Contains per record metric
        self.metric_names = ['MeanSquared']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Compute the msq error metric
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y
        Returns:
            float: Returns the metric
        """
        # compute error
        msq=self.be.square(y - t) / 2.
        self.outputs=msq.astensor()

        return self.outputs.get()[:, calcrange].mean()


class SmoothL1Metric(Metric):

    """
    Compute the SmoothL1Metric
    """

    def __init__(self):
        self.outputs = self.be.iobuf(1)  # Contains per record metric
        self.metric_names = ['SmoothL1']

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Compute the smooth-L1 error metric
        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y
        Returns:
            float: Returns the metric
        """
        # compute error
        sl1=SmoothL1Loss()
        self.outputs=sl1.smoothL1(y-t).astensor()

        return self.outputs.get()[:, calcrange].mean()
        