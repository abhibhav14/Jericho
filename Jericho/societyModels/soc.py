import numpy as np
import numpy.linalg as la
import math
import random
from tqdm import tqdm

import os
from time import sleep

from jericho import Jericho

class Society(object):
    """
    Represents a society
    """

    def __init__(self, numExplore, numExploit, world):
