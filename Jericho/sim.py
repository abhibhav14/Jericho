from jericho import Jericho
from agentModels.exploreExploit import Agent
import random
import os
from time import sleep
import numpy as np
import utils

from tqdm import tqdm

import matplotlib.pyplot as plt

memsizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500]


memlen = 4
newProb = 0.05

g = list()
p = list()
l = list()
c = 0

numEp = 100
seeds = np.arange(numEp)

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
        agents.append(Agent(random.randrange(int(world.size/3)), random.randrange(int(world.sizeW/3)), world, memsize=memlen))
        c += 1

    random.seed(seeds[15])
    np.random.seed(seeds[15])
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
                        agents.append(Agent(pos[0], pos[1], world, created=j, memsize=memlen, explore = agentTemp[i].explore))
                        c += 1

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

# print(np.mean(g))
# print(len(np.nonzero(g)[0]))
# print(sum(g) / len(np.nonzero(g)[0]))

print("Percentage that lived entire life", np.mean(p))
# print(len(np.nonzero(p)[0]))
# print(sum(p) / len(np.nonzero(p)[0]))

print("Median Life", np.median(l))
print("Mean Life", np.mean(l))
print("Life Gini", utils.compute_gini(l))
plt.hist(l, bins=30)
# plt.show()
print("Number of Agents", c)
