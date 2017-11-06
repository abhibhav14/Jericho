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


    def __init__(self, xStart, yStart, world, memory=None, memsize=3, created=0):

        world.addAgent(self, xStart, yStart)
        self.world = world

        self.memLength = memsize
        if memory is None:
            self.memory = list()
        else:
            self.memory = memory

        self.created = created
        self.energy = 45
        self.revisit = 10
        self.livingCost = 3
        self.memCost = 0.5
        self.maxEnergy = 100
        self.wealth = 0
        self.life = np.random.poisson(100)

        self.confidence = 8
        self.confAdd = 1
        self.cFlag = False
        self.remProb = 0.75

        self.alive = True
        self.range = 5

        self.createLevel = 80

        self.communication = 2
        self.commProb = 1
        self.commEnergy = 30
        self.countDown = 2
        self.countDownInit = 2

        self.lie = True
        self.noF = None

    def _dist_from(self, x, y):
        v = [self.attr.i - x, self.attr.j - y]
        return la.norm(v, 1)

    def step(self):
        self.life -= 1
        self.countDown -= 1
        if self.life == 0:
            self.alive = False
            return -1
        if not self.alive:
            return 0
        see, seeAgents = self.world.sense(self)
        # if len(seeAgents) > 0:
            # print("sees")
            # print(self.attr.id, seeAgents)
            # input()
            # pass
        eC, eG = 0, 0

        if len(seeAgents) != 0 and self.energy < self.commEnergy and len(self.memory) == 0 and self.countDown <= 0:
            self.countDown += self.countDownInit
            agComm = random.choice(seeAgents)
            if len(agComm.memory) > 0:
                for i in np.arange(agComm.communication):
                    if self.lie:
                        if self.noF is not None:
                            self.memoryProcess([noF[0], noF[1], 12, 32, True], True)
                    elif random.random() < self.commProb:
                        self.memoryProcess(random.choice(agComm.memory), True)


        if len(see) == 0: # No food, explore or go to memory
            if self.noF is None and len(self.memory) > 0:
                if min([self._dist_from(i[0], i[1]) for i in self.memory]) > 5:
                    self.noF = [self.attr.i, self.attr.j]

            uMem = [[self._dist_from(i[0], i[1]), num] for num, i in enumerate(self.memory) if i[2] == 0]
            if len(uMem) == 0:
                range = self.range
                if self.energy < 0:
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
                x, y, kn, c, t = self.memory[k]
                x = max(0, min(self.world.size - 1, x + random.randrange(-1, 2)))
                y = max(0, min(self.world.sizeW - 1, y + random.randrange(-1, 2)))
                dist = self._dist_from(x, y)
                eC = int(dist * self.memCost)
                eG = self.world.step(self, x, y)
                if eG == 0:
                    self.memory[k][3] /= 2
                else:
                    self.memory[k][3] = self.confidence + self.confAdd
                    self.memory[k][4] = True
                self.memory.append(self.memory.pop(k))

        else: # If you see food, go there
            d = [self._dist_from(i, j) for i, j in see]
            x = see[np.argmin(d)]
            eC = np.argmin(d)
            eG = self.world.step(self, x[0], x[1])

            # Memory stuff
            if eG > 4:
                self.memoryProcess(x)

        eC = eC / 2
        eC += self.livingCost
        if self.energy - eC <= 0:
            self.alive = False
            return 0
        if self.energy + eG - eC > 100:
            diff = 100 - self.energy
            self.wealth += eG - eC + diff
            self.energy = 100
        else:
            self.energy = min(self.energy + eG - eC, self.maxEnergy)

        for i in self.memory:
            i[2] = max(0, i[2] - 1)

        if self.energy > self.createLevel:
            return 2
        else:
            return 1

    def memoryProcess(self, x, agentSay = False):
        if len(x) == 5:
            if not x[4]:
                return
        if len(self.memory) == 0:
            self.memory.append([x[0], x[1], self.revisit, self.confidence, False])
        if min([self._dist_from(i[0], i[1]) for i in self.memory]) < 5:
            return
        if len(self.memory) < self.memLength:
            self.memory.append([x[0], x[1], self.revisit, self.confidence, False])
        else:
            if random.random() > self.remProb or agentSay:
                ci = [frag[3] for frag in self.memory]
                ci = np.max(ci) - ci + 1e-6
                ci = ci / np.sum(ci)
                rem = np.argmax(np.random.multinomial(1, ci))
                if (agentSay and self.memory[rem][3] < x[3]):
                    self.memory.pop(rem)
                    self.memory.insert(0, [x[0], x[1], 0, x[3], False])
                else:
                    self.memory[rem] = [x[0], x[1], self.revisit, self.confidence, False]
                return True
