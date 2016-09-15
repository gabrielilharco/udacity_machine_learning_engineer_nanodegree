import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
       
        self.Q_learner = {}
        self.reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)

        # Select action according to your policy
        possible_actions = {}
        for action in self.env.valid_actions:
            possible_actions[action] = self.Q_learner.get((self.state, action), 0)
        max_utility = max(possible_actions.values())

        best_actions = []
        for action in self.env.valid_actions:
            if possible_actions[action] == max_utility:
                best_actions.append(action)
        best_action = random.choice(best_actions)

        # Execute action and get reward
        reward = self.env.act(self, best_action)
        self.reward += reward

        # Learn policy based on state, action, reward
        alpha = 1.0
        self.Q_learner [(self.state, best_action)] = \
            (1-alpha)*self.Q_learner.get((self.state, best_action),0) + \
            alpha*reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    n_trials = 10000
    sim.run(n_trials=n_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print "Successful run rate: {}/{} = {}".format(sim.successful_runs, n_trials, sim.successful_runs / float(n_trials))

if __name__ == '__main__':
    run()
