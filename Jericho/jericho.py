import numpy as np
import numpy.linalg as la
import math
import random
# import scipy as sp
import tqdm as tqdm

class Property(object):

    def __init__(self, i, j, id):
        self.i = i
        self.j = j
        self.id = id

class Jericho(object):
    """
    Class for the environment
    for one agent
    0 - nothing
    1 - agent
    2 - sparse food
    3 - lot of food
    """

    def __init__(self, size=60, sizeW=100):

        self.denseStats = 0
        self.spareStats = 0
        
        self.size = size
        self.sizeW = sizeW
        self.world = np.zeros((self.size, self.sizeW))
        self.time = 0

        self.sourceNum = int((self.size * self.sizeW) ** 0.25) * 3

        self.sources = np.zeros((self.sourceNum, 3))
        for k in range(self.sourceNum):
            i, j = random.randrange(self.size - 1), random.randrange(self.sizeW - 1)
            d = random.choice([0.5, 0.5, 0.5])
            self.sources[k] = [i, j, d]

        self.sparseCount = 0
        self.sparsity = 2
        self.idealsparseCount = self.size * self.sizeW * self.sparsity * 0.01

        self._reset_world()

        self.agentList = list()

        self.idCount = 1
        self.denseAmout = 10


    def _reset_world(self):
        self.world = np.zeros((self.size, self.sizeW))
        self.time = 0
        for iF, jF, dF in self.sources:
            i  = int(iF)
            j  = int(jF)
            self.world[i][j] = -9
            self.world[i+1][j] = -9
            self.world[i-1][j] = -9
            self.world[i][j+1] = -9
            self.world[i][j-1] = -9

        for i in range(self.size):
            for j in range(self.sizeW):
                if random.random() < self.sparsity * 0.01:
                    self.world[i][j] = -8
                    self.sparseCount += 1

    def _print_world(self):
        for i in range(self.size):
            print("|", end="")
            for j in range(self.sizeW):
                if self.world[i][j] == 0:
                    print(" ", end="|")
                elif self.world[i][j] > -1:
                    ag = self.agentList[int(self.world[i][j] - 1)]
                    print('\x1b[6;30;44m' + str(len(ag.memory)) + '\x1b[0m', end="|")
                    # if ag.alive:
                        # print('\x1b[6;30;44m' + str(int(self.world[i][j])) + '\x1b[0m', end="|")
                        # print('\x1b[6;30;44m' + 'a' + '\x1b[0m', end="|")
                    # else:
                        # print('\x1b[6;30;41m' + str(int(self.world[i][j])) + '\x1b[0m', end="|")
                        # print('\x1b[6;30;41m' + 'a' + '\x1b[0m', end="|")
                elif self.world[i][j] < -7:
                    print(int(np.abs(self.world[i][j])), end="|")
            print()

    def _in_bounds(self, i, j):
        if i >= 0 and j >= 0 and i < self.size and j < self.sizeW:
            return True
        return False

    def _dist_from(self, agent, x, y):
        v = [agent.attr.i - x, agent.attr.j - y]
        return la.norm(v, 1)

    def addAgent(self, ag, i, j):
        prop = Property(i, j, self.idCount)
        self.idCount += 1
        ag.attr = prop
        self.agentList.append(ag)
        self.world[i][j] = ag.attr.id

    def removeAgent(self, ag):
        self.world[ag.attr.i][ag.attr.j] = 0

    def findEmpty(self, ag):
        ai, aj = ag.attr.i, ag.attr.j
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if self._in_bounds(ai+i, aj+j) and self.world[ai+i][aj+j] == 0:
                    return ai+i, aj+j
        return None

    def sense(self, a):
        ai = None
        aj = None

        ai, aj = a.attr.i, a.attr.j

        s = list()
        ag = list()
        for i in range(-6, 7):
            for j in range(-6, 7):
                if self._in_bounds(ai+i, aj+j) and self.world[ai+i][aj+j] < -7 and i in range(-2, 3) and j in range(-2, 3):
                    s.append([ai+i, aj+j])
                elif self._in_bounds(ai+i, aj+j) and self.world[ai+i][aj+j] != a.attr.id and self.world[ai+i][aj+j] > 0:
                    ag.append(self.agentList[int(self.world[ai+i][aj+j]-1)])
        s = np.array(s)
        ag = np.array(ag)
        if len(s) == 0:
            return s, ag

        return s[np.random.choice(range(len(s)), min(3, len(s)), replace=False)], ag

    def step(self, ag, i, j):

        energy = 0
        if self.world[i][j] == -8:
            self.sparseCount -= 1
            energy = 4
            self.spareStats += 4
        elif self.world[i][j] == -9:
            energy = self.denseAmout
            self.denseStats += 7
        elif self.world[i][j] != 0:
            return 0

        self.world[ag.attr.i][ag.attr.j] = 0
        ag.attr.i = i
        ag.attr.j = j
        self.world[ag.attr.i][ag.attr.j] = ag.attr.id
        return energy

    def stepWorld(self):
        # With high probability regrow the food in the sources
        self.time += 1
        if self.time % 10 == 0:
            for iF, jF, dF in self.sources:
                i  = int(iF)
                j  = int(jF)
                self.world[i][j] = -1 * min(9, -1 * self.world[i][j] + 9 * int(random.random() < dF))
                self.world[i+1][j] = -1 * min(9, -1 * self.world[i+1][j] + 9 * int(random.random() < dF))
                self.world[i-1][j] = -1 * min(9, -1 * self.world[i-1][j] + 9 * int(random.random() < dF))
                self.world[i][j+1] = -1 * min(9, -1 * self.world[i][j+1] + 9 * int(random.random() < dF))
                self.world[i][j-1] = -1 * min(9, -1 * self.world[i][j-1] + 9 * int(random.random() < dF))

        # if self.sparseCount < int((self.sparsity - 1) * self.size * self.sizeW * 0.01):
        if self.time % 50 == 0:
            if self.sparseCount > self.idealsparseCount:
                pass
            else:
                num = np.random.poisson(self.idealsparseCount - self.sparseCount)
                while num > 0:
                    num -= 1
                    while True:
                        i, j = random.randrange(0, self.size), random.randrange(0, self.sizeW)
                        if self.world[i][j] == 0:
                            self.world[i][j] = -8
                            self.sparseCount += 1
                            break
                        else:
                            pass
