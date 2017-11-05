from jericho import Jericho
from agentModels.agentFixedForgetComm import Agent
import random
import os
from time import sleep
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

memsizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500]


memlen = 4
arr = list()
arr2 = list()
newProb = 0.1

for episodes in tqdm(range(50)):

    world = Jericho(60, 100)
    agents = list()
    agentTemp = list()
    lifetimes = list()
    for i in range(10):
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
                        agents.append(Agent(pos[0], pos[1], world, created=j, memory=agentTemp[i].memory[:], memsize=memlen))

            else:
                lifetimes.append(j - agentTemp[i].created)
                agents.remove(agentTemp[i])
                world.removeAgent(agentTemp[i])
        world.stepWorld()
        os.system('clear')
        world._print_world()
        print(len(agents), j)
        sleep(0.15)
        if len(agents) == 0:
            break
        if j == 999:
            break

    print(world.spareStats)
    print(world.denseStats)
    arr.append(world.spareStats)
    arr2.append(world.denseStats)

plt.plot(arr, 'r')
plt.plot(arr2, 'b')
plt.show()
