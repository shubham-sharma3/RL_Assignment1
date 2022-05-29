import numpy as np
from agent import Agent


# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # Policy initialization
        # ******************
        # 1.1.a)

        self.V = {s: 0 for s in states}

        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        counter = 0

        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {}
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # 1.1.b)
                    if len(self.mdp.getPossibleActions(s)) == 0:
                        newV[s] = 0
                    else:
                        newValue = 0
                        for stateAndProb in mdp.getTransitionStatesAndProbs(s, a):
                            newValue += stateAndProb[1] * (mdp.getReward(s, a, stateAndProb[0]) 
                                                            + self.discount * self.V[stateAndProb[0]])
                        newV[s] = newValue


                # update value estimate
                self.V = newV

                # ******************

            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # 1.1.c)
                    maxA = None
                    maxQ = None
                    for a in actions:
                        update = maxQ == None
                        q = 0
                        for stateAndProb in mdp.getTransitionStatesAndProbs(s, a):
                            q += stateAndProb[1] * (mdp.getReward(s, a, stateAndProb[0]) 
                                                            + self.discount * self.V[stateAndProb[0]])
                        if(not update):
                            update = q > maxQ
                        if(update):
                            maxQ = q
                            maxA = a
                    self.pi[s] = maxA
                    policy_stable = policy_stable and (old_action == maxA)

                    # ****************
            counter += 1
            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # 1.2.
        return self.V[state]

        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # 1.3.
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
        # **********
        # 1.4.

        return self.pi[state]

        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
