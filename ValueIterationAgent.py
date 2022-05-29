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
        # 2.1 a)
        self.V = {s: 0 for s in states}


        # ************

        for i in range(iterations):
            newV = {}
            updateInPlace = False
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # 2.1. b)
                if len(actions) == 0:
                    if updateInPlace:
                        self.V[s] = 0
                    else:
                        newV[s] = 0
                else:
                    qValues = {}
                    for a in actions:
                        q = 0
                        for stateAndProb in mdp.getTransitionStatesAndProbs(s, a):
                            q += stateAndProb[1] * (mdp.getReward(s, a, stateAndProb[0]) 
                                                            + self.discount * self.V[stateAndProb[0]])
                        qValues[a] = q
                    maxQ = qValues[max(qValues, key=lambda k: qValues[k])]
                    if updateInPlace:
                        self.V[s] = maxQ
                    else:
                        newV[s] = maxQ

            # Update value function with new estimate
            if(not updateInPlace):
                self.V = newV

            # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # 2.2
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
        # 2.3.
        q = 0
        for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):
            q += stateAndProb[1] * (self.mdp.getReward(state, action, stateAndProb[0]) 
                                            + self.discount * self.V[stateAndProb[0]])
        return q
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
            #  2.4
            # Evaluate Q-Values
            qValues = {a: self.getQValue(state, a) for a in actions}
            maxAction = max(qValues, key=lambda k: qValues[k])
            return maxAction
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
