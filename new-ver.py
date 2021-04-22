from numpy import array, sum, exp, average
from numpy.random import seed, rand


A = lambda x: x * (x > 0)
m = lambda x: 1 * (x > 0)
norm = lambda x: exp(e) / sum(exp(e))


class nn:

    def __init__(self, shape):
        self.links = array([*map(rand, shape[:-1], shape[1:])])


    def forward(self, x):

        for i, links in enumerate(self.links):
            x.append(A(x[i] @ links))
         
        norm(x)


    def back(self, x, y):
      pass


    def fit(self, x, y, rate=.1):

        self.forward(x)
        self.back(x, y)
        self.links -= rate * self.change


    def predict(self, x):
        self.forward(x)
        return x[-1]
