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
        self.V = {s: 0 for s in states}

        # ************

        for i in range(iterations):
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                maxValue = 0
                if len(self.mdp.getPossibleActions(s)) == 0:
                    newV[s] = 0
                else:
                    # TODO need sum over actions here? will pi get probabilistic later?
                    for action in actions:
                        val = 0
                        tranisitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(
                            s, action)

                        for tranisitionStatesAndProb in tranisitionStatesAndProbs:
                            discountedValue = self.discount * \
                                self.V[tranisitionStatesAndProb[0]]

                            reward = self.mdp.getReward(
                                s, action, tranisitionStatesAndProb[0])

                            val += reward + \
                                (discountedValue * tranisitionStatesAndProb[1])

                        maxValue = max(maxValue, val)

                    newV[s] = maxValue

            # update value estimate
            self.V = newV

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
            # TODO 2.4

            Q = {}

            for action in actions:
                q = 0
                for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):
                    q = stateAndProb[1] * (self.mdp.getReward(state, action, stateAndProb[0])
                                           + self.discount * self.V[stateAndProb[0]])
                Q[action] = q

            return max(Q, key=Q.get)

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
