from numpy import array, zeros, ones, exp, average, newaxis
from numpy.random import randn


def A(x):
    return 1 / (1 + exp(-x))


X = array([
    [0, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 1]
])


Y = array([[1, 0, 1, 0, 1, 0, 0, 0]]).T


class nn:
    def __init__(self, *shape):
        self.links = array([*map(randn, shape[:-1], shape[1:])], dtype=object)
        

    def forward(self, x):
        outs = [ x ]

        for i, links in enumerate(self.links):
            outs.append(A(net[i] @ links))

        return outs


    def back(self, net, y):
        loss = net[-1] - y
        yield average(net[-2][:, :, newaxis] * loss[:, newaxis, :], axis=0)

        for i, layer in reversed(list(enumerate(net))[1:-1]):

            loss = layer * (1 - layer) * (loss @ self.links[i].T)
            prev = net[i - 1]

            yield average(prev[:, :, newaxis] * loss[:, newaxis, :], axis=0)


    def fit(self, x, y, time):

        for i in range(time):
          self.links += -0.5 * array(list(reversed(list(self.back(self.forward(x), y)))), dtype=object)


    def predict(self, x):
        return self.forward(x)[-1]
