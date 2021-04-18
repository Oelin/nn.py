from numpy import array, exp, average, newaxis
from numpy.random import randn


def A(x):
    return 1 / (1 + exp(-x))


X = array([
  [0, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [0, 1, 1, 0],
  [1, 0, 0, 0],
  [1, 0, 1, 0],
  [1, 1, 0, 0],
  [1, 1, 1, 0],
  [1, 0, 0, 1]
])


Y = array([[0, 0, 0, 1, 0, 1, 1, 1, 1]]).T


class nn:
    def __init__(self, *shape):
        self.edges = list(map(randn, shape[:-1], shape[1:]))


    def forward(self, x):
        net = [ x ]

        for i, edges in enumerate(self.edges[:-1]):
            net.append(A(net[i] @ edges))

        net.append(net[-1] @ self.edges[-1])
        return net


    def back(self, net, y):
        loss = net[-1] - y
        yield average(net[-2][:, :, newaxis] * loss[:, newaxis, :], axis=0)

        for i, layer in reversed(list(enumerate(net))[1:-1]):

            loss = layer * (1 - layer) * (loss @ self.edges[i].T)
            prev = net[i - 1]

            yield average(prev[:, :, newaxis] * loss[: , newaxis, :], axis=0)


    def learn(self, x, y, time):

        for i in range(time):
          deltas = list(reversed(list(self.back(self.forward(x), y))))

          for j, delta in enumerate(deltas):
              self.edges[j] += -0.5 * delta