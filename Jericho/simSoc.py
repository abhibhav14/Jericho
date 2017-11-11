from jericho import Jericho
from agentModels.exploreExploit import Agent
import random
import os
from time import sleep
import time
import numpy as np
import utils
import numpy.linalg as la

from tqdm import tqdm

import matplotlib.pyplot as plt

memsizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500]


memlen = 4
newProb = 0.05
switchProb = 0.2

g = list()
p = list()
l = list()
lexp = list()
exp = list()
c = 0

numEp = 100
seeds = np.arange(numEp)

def dist_from(x, y, tox=None, toy=None):
    if tox is None:
        return -1
    else:
        v = [tox - x, toy - y]
        return la.norm(v, 1)

for episodes in tqdm(range(1)):

    random.seed(seeds[20])
    np.random.seed(seeds[20])
    world = Jericho(60, 100)
    agents = list()
    agentTemp = list()
    wealth = list()
    countdead = 0
    countnatural = 0
    lifetimes = list()

    for i in range(5):
        agents.append(Agent(random.randrange(int(world.size/3)), random.randrange(int(world.sizeW/3)), world, memsize=memlen, explore=True))
        c += 1
    for i in range(5):
        agents.append(Agent(random.randrange(int(world.size/3)), random.randrange(int(world.sizeW/3)), world, memsize=memlen, explore=False))

    random.seed(seeds[15])
    np.random.seed(seeds[15])
    for j in tqdm(range(5000)):
        agentTemp = agents[:]
        for i in np.random.permutation(np.arange(len(agentTemp))):
            n = agentTemp[i].step()
            if n == 1:
                pass
            elif n == 2:
                if random.random() < newProb:
                    pos = world.findEmpty(agentTemp[i])
                    if pos is not None:
                        if random.random() < switchProb:
                            agents.append(Agent(pos[0], pos[1], world, created=j, memsize=memlen, explore = not agentTemp[i].explore))
                        else:
                            agents.append(Agent(pos[0], pos[1], world, created=j, memsize=memlen, explore = agentTemp[i].explore))
                        c += 1

            else:
                if n == -1:
                    wealth.append(agentTemp[i].wealth)
                    countnatural += 1
                agents.remove(agentTemp[i])
                countdead += 1
                world.removeAgent(agentTemp[i])
                if not agentTemp[i].explore:
                    l.append(j - agentTemp[i].created)
                else:
                    lexp.append(j - agentTemp[i].created)
        world.stepWorld()
        # os.system('clear')
        # world._print_world()
        # print(len(agents), j)
        # sleep(0.25)
        if len(agents) == 0:
            break
        c1 = 0
        for i in agents:
            if i.explore:
                c1 += 1
        exp.append(c1 * 100 / len(agents))
        if j % 50 == 55:
            # print(len(agentTemp))
            # print(np.sum([len(j.memory) for j in agentTemp]))
            spots = list()
            totalEnergy = 0
            for i in agentTemp:
                totalEnergy += (i.energy) * (i.energy > 0)
                for j in i.memory:
                    if len(spots) == 0:
                        spots.append(j)
                    else:
                        if np.min([dist_from(j[0], j[1], m[0], m[1]) for m in spots]) > 5:
                            spots.append(j)

            # Redistribute them acc to spots
            # print(totalEnergy, len(agents))
            numexplorer = c1
            numexploiter = len(agents) - c1
            # print(numexplorer)
            # print(numexploiter)
            explorerPerc = 0.65
            if len(spots) > 0:
                for i in agents:
                    if i.explore and numexplorer:
                        i.energy = (explorerPerc) * (totalEnergy) * (1 / numexplorer)
                    elif not i.explore and numexploiter:
                        i.energy = (1 - explorerPerc) * (totalEnergy) * (1 / numexploiter)
                    if not i.explore:
                        m = random.choice(spots)
                        i.memory = list()
                        i.memoryProcess(m, True)

                        world.step(i, m[0], m[1])
    g.append(utils.compute_gini(wealth))
    # l.append(np.mean(lifetimes))
    p.append(countnatural / countdead)

print(np.mean(g))
# print(len(np.nonzero(g)[0]))
# print(sum(g) / len(np.nonzero(g)[0]))

print("Percentage that lived entire life", np.mean(p))
# print(len(np.nonzero(p)[0]))
# print(sum(p) / len(np.nonzero(p)[0]))

print("for exploiter")
print("Median Life", np.median(l))
print("Mean Life", np.mean(l))
print("Life Gini", utils.compute_gini(l))

print("for explorer")
print("Median Life", np.median(lexp))
print("Mean Life", np.mean(lexp))
print("Life Gini", utils.compute_gini(lexp))

print("combined")
print("Median Life", np.median(lexp + l))
print("Mean Life", np.mean(lexp + l))
print("Life Gini", utils.compute_gini(lexp + l))
# plt.hist(l, bins=30)
# plt.show()
print("Number of Agents", c)
plt.plot(exp)
# plt.show()
