import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        for k in self.Q.keys():
           if state == k[0]:
              a = self.getPolicy(state)
              return self.getQValue(state, a)

        return 0.0
        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        if (state, action) in self.Q.keys():
           return self.Q[(state, action)]    
       
        return 0.0
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        a_map = {}
        for k in self.Q.keys():
           if state == k[0]:
              a_map[k[1]] = self.Q[k]
        if len(a_map) > 0:
           return max(a_map, key=a_map.get)

        return self.getRandomAction(state)
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            return np.random.choice(all_actions)
            # *********
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        if np.random.rand() < self.epsilon:
           return self.getRandomAction(state)
        else:
           return self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        if state != (-1,-1):
           if (state, action) not in self.Q.keys():
              self.Q[(state, action)] = 0.0

           nextA_map = {}
           for k in self.Q.keys():
              if nextState == k[0]:
                 nextA_map[k[1]] = self.Q[k]
      
           if len(nextA_map) < 1:
              Q_max = 0.0
           else:     
              Q_max = max(nextA_map.values())
           self.Q[(state, action)] = self.Q[(state, action)]*(1-self.learningRate) + self.learningRate * (reward + self.discount*(Q_max))
        # *********
