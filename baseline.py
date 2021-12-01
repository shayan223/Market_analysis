import numpy as np
import pandas as pd


'''A baseline random agent, that arbitrarily buys and sells random amounts'''

class baseline_agent:

    def __init__(self, buy_amount):
        self.reward = 0

        #Increments of how much we are buying/selling
        self.buying_power = buy_amount

        #Keeps track of reward history
        self.history = []

    def take_action(self):
        #50/50 chance of buy sell
        buy_choice = np.random.randint(0,1)




