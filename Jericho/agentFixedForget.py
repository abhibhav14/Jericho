import numpy as np
import numpy.linalg as la
import math
import random
from tqdm import tqdm

import os
from time import sleep

from jericho import Jericho

class Agent(object):
    """
    Represents an Agent
    """


    def __init__(self, xStart, yStart, world, memory=None, created=0):

        world.addAgent(self, xStart, yStart)
        self.world = world
        if memory is None:
            self.memory = list()
        else:
            self.memory = memory

        self.created = created
        self.energy = 45
        self.revisit = 15
        self.livingCost = 1
        self.memCost = 0.5
        self.maxEnergy = 100
        self.life = np.random.poisson(150)

        self.confidence = 10000
        self.cFlag = False

        self.alive = True

        self.createLevel = 80

    def _dist_from(self, x, y):
        v = [self.attr.i - x, self.attr.j - y]
        return la.norm(v, 1)

    def step(self):
        self.life -= 1
        if self.life == 0:
            self.alive = False
        if not self.alive:
            return 0
        see, seeAgents = self.world.sense(self)
        # if len(seeAgents) > 0:
            # print("sees")
            # print(self.attr.id, seeAgents)
            # input()
            # pass
        eC, eG = 0, 0

        if len(see) == 0: # No food, explore or go to memory
            uMem = [[self._dist_from(i[0], i[1]), num] for num, i in enumerate(self.memory) if i[2] == 0]
            if len(uMem) == 0:
                range = int(0.25 * self.energy)
                if range == 0:
                    eC = 3 # Hack to just kill the agent, does not mean anything
                else:
                    i, j = random.randrange(-range, range + 1), random.randrange(-range, range + 1)
                    i = max(0, min(i + self.attr.i, self.world.size - 1))
                    j = max(0, min(j + self.attr.j, self.world.sizeW - 1))
                    eC = self._dist_from(i, j)
                    self.world.step(self, i, j)
            else:
                k = np.argmin(uMem, axis=0)[0]
                k = uMem[k][1]
                self.memory[k][2] = self.revisit
                x, y, kn, c = self.memory[k]
                x = max(0, min(self.world.size - 1, x + random.randrange(-1, 2)))
                y = max(0, min(self.world.sizeW - 1, y + random.randrange(-1, 2)))
                dist = self._dist_from(x, y)
                eC = int(dist * self.memCost)
                eG = self.world.step(self, x, y)
                if eG == 0:
                    self.memory[k][3] -= 1
                    if self.memory[k][3] == 0:
                        self.memory.remove(self.memory[k])
                else:
                    self.memory[k][3] = self.confidence

        else: # If you see food, go there
            d = [self._dist_from(i, j) for i, j in see]
            x = see[np.argmin(d)]
            eC = np.argmin(d)
            eG = self.world.step(self, x[0], x[1])

            # Memory stuff
            if eG > 4:
                if len(self.memory) == 0:
                    self.memory.append([x[0], x[1], self.revisit, self.confidence])
                elif len(self.memory) < 3:
                    if min([self._dist_from(i, j) for i, j, k, l in self.memory]) < 5:
                        pass
                    else:
                        self.memory.append([x[0], x[1], self.revisit, self.confidence])
                else:
                    pass

        eC = eC / 2
        eC += self.livingCost
        if self.energy - eC <= 0:
            self.alive = False
            return 0
        self.energy = min(self.energy + eG - eC, self.maxEnergy)
        for i in self.memory:
            i[2] = max(0, i[2] - 1)

        if self.energy > self.createLevel:
            return 2
        else:
            return 1
