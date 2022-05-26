from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        self.V = {s: 0.0 for s in states}
        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}
        # ************
        counter = 0
        for i in range(iterations):
            newV = {}
            self.pi = {}
            pref_a = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                if len(actions) < 1:
                   newV[s] = 0.0
                else:
                   q_neu = {}
                   #newV[s] = 0.0
                   for a in actions:
                      q_neu[a] = self.getQValue(s, a)
                   newV[s] = max(q_neu.values())

                # Update value function with new estimate
                self.V[s] = newV[s]        
                # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]
        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        q_val = 0.0
        for sp, prob in self.mdp.getTransitionStatesAndProbs(state, action):
           q_val += prob * (self.mdp.getReward(state, action, None) + self.discount * self.V[sp])
        return q_val
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
           return None

        else:
        # **********
        # TODO 2.4
          Q = {}
          for a in actions:
             Q[a] = self.getQValue(state, a)
          self.pi[state] = max(Q, key=Q.get)
          return self.pi[state]
        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
