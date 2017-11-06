from jericho import Jericho
from agentModels.agentFixedForgetComm import Agent
import random
import os
from time import sleep
import numpy as np
import utils

from tqdm import tqdm

import matplotlib.pyplot as plt

memsizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500]


memlen = 2
newProb = 0.1

g = list()
p = list()
l = list()

for episodes in tqdm(range(100)):

    world = Jericho(60, 100)
    agents = list()
    agentTemp = list()
    wealth = list()
    countdead = 0
    countnatural = 0
    lifetimes = list()

    for i in range(5):
        agents.append(Agent(random.randrange(int(world.size/3)), random.randrange(int(world.sizeW/3)), world, memsize=memlen))

    for j in range(1000):
        agentTemp = agents[:]
        for i in np.random.permutation(np.arange(len(agentTemp))):
            n = agentTemp[i].step()
            if n == 1:
                pass
            elif n == 2:
                if random.random() < newProb:
                    pos = world.findEmpty(agentTemp[i])
                    if pos is not None:
                        agents.append(Agent(pos[0], pos[1], world, created=j, memsize=memlen))

            else:
                if n == -1:
                    wealth.append(agentTemp[i].wealth)
                    countnatural += 1
                agents.remove(agentTemp[i])
                countdead += 1
                world.removeAgent(agentTemp[i])
                l.append(j - agentTemp[i].created)
        world.stepWorld()
        # os.system('clear')
        # world._print_world()
        # print(len(agents), j)
        # sleep(0.25)
        if len(agents) == 0:
            break
        if j == 999:
            break
    g.append(utils.compute_gini(wealth))
    # l.append(np.mean(lifetimes))
    p.append(countnatural / countdead)

print(np.mean(g))
# print(len(np.nonzero(g)[0]))
# print(sum(g) / len(np.nonzero(g)[0]))

print(np.mean(p))
# print(len(np.nonzero(p)[0]))
# print(sum(p) / len(np.nonzero(p)[0]))

print(np.median(l))
plt.hist(l, bins=30)
plt.show()
