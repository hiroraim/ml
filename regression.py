import numpy as np

class LinearRegression:
    """
    >>> import regression
    >>> model = regression.LinearRegression()
    >>> model.x
    >>> #nothing
    """
    x = None
    theta = None
    y = None

    def fit(self, x, y):
        """
        >>> import importlib
        >>> import regression
        >>> import main
        >>> X,Y = main.load_linear_example1()
        >>> importlib.reload(regression)
        <module 'regression' from '/Users/e175765/ml/regression.py'>
        >>> model = regression.LinearRegression()
        >>> model.fit(X,Y)
        >>> model.theta
        array([5.30412371, 0.49484536])
        """
        temp = np.linalg.inv(np.dot(x.T,x))
        self.theta = np.dot(np.dot(temp,x.T),y)

    def predict(self, x):
        pass

    def score(self, x, y):
        pass