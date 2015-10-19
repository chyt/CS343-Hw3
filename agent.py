from OpenNero import *
from common import *
import random
import math

import Maze
from Maze.constants import *
import Maze.agent
from Maze.agent import *

class MyTabularRLAgent(AgentBrain):
    """
    Tabular RL Agent
    """
    def __init__(self, gamma, alpha, epsilon):
        """
        Constructor that is called from the robot XML file.
        Parameters:
        @param gamma reward discount factor (between 0 and 1)
        @param alpha learning rate (between 0 and 1)
        @param epsilon parameter for the epsilon-greedy policy (between 0 and 1)
        """
        AgentBrain.__init__(self) # initialize the superclass
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        """
        Our Q-function table. Maps from a tuple of observations (state) to
        another map of actions to Q-values. To look up a Q-value, call the predict method.
        """
        self.Q = {} # our Q-function table
        print

    def __str__(self):
        return self.__class__.__name__ + \
            ' with gamma: %g, alpha: %g, epsilon: %g' \
            % (gamma, alpha, epsilon)

    def initialize(self, init_info):
        """
        Create a new agent using the init_info sent by the environment
        """
        self.action_info = init_info.actions
        self.sensor_info = init_info.sensors
        return True

    def predict(self, observations, action):
        """
        Look up the Q-value for the given state (observations), action pair.
        """
        o = tuple([x for x in observations])
        if o not in self.Q:
            return 0
        else:
            return self.Q[o][action]

    def update(self, observations, action, new_value):
        """
        Update the Q-function table with the new value for the (state, action) pair
        and update the blocks drawing.
        """
        o = tuple([x for x in observations])
        actions = self.get_possible_actions(observations)
        if o not in self.Q:
            self.Q[o] = [0 for a in actions]
        self.Q[o][action] = new_value
        self.draw_q(o)

    def draw_q(self, o):
        e = get_environment()
        if hasattr(e, 'draw_q'):
            e.draw_q(o, self.Q)

    def get_possible_actions(self, observations):
        """
        Get the possible actions that can be taken given the state (observations)
        """
        aMin = self.action_info.min(0)
        aMax = self.action_info.max(0)
        actions = range(int(aMin), int(aMax+1))
        return actions

    def get_max_action(self, observations):
        """
        get the action that is currently estimated to produce the highest Q
        """
        actions = self.get_possible_actions(observations)
        max_action = actions[0]
        max_value = self.predict(observations, max_action)
        for a in actions[1:]:
            value = self.predict(observations, a)
            if value > max_value:
                max_value = value
                max_action = a
        return (max_action, max_value)

    def get_epsilon_greedy(self, observations, max_action = None, max_value = None):
        """
        get the epsilon-greedy action
        """
        actions = self.get_possible_actions(observations)
        if random.random() < self.epsilon: # epsilon of the time, act randomly
            return random.choice(actions)
        elif max_action is not None and max_value is not None:
            # we already know the max action
            return max_action
        else:
            # we need to get the max action
            (max_action, max_value) = self.get_max_action(observations)
            return max_action

    def start(self, time, observations):
        """
        Called to figure out the first action given the first observations
        @param time current time
        @param observations a DoubleVector of observations for the agent (use len() and [])
        """
        self.previous_observations = observations
        self.previous_action = self.get_epsilon_greedy(observations)
        return self.previous_action

    def act(self, time, observations, reward):
        """
        return an action given the reward for the previous action and the new observations
        @param time current time
        @param observations a DoubleVector of observations for the agent (use len() and [])
        @param the reward for the agent
        """
        # get the reward from the previous action
        r = reward[0]

        # get the updated epsilon, in case the slider was changed by the user
        self.epsilon = get_environment().epsilon

        # get the old Q value
        Q_old = self.predict(self.previous_observations, self.previous_action)

        # get the max expected value for our possible actions
        (max_action, max_value) = self.get_max_action(observations)

        # update the Q value
        self.update( \
            self.previous_observations, \
            self.previous_action, \
            Q_old + self.alpha * (r + self.gamma * max_value - Q_old) )

        # select the action to take
        action = self.get_epsilon_greedy(observations, max_action, max_value)
        self.previous_observations = observations
        self.previous_action = action
        return action

    def end(self, time, reward):
        """
        receive the reward for the last observation
        """
        # get the reward from the last action
        r = reward[0]
        o = self.previous_observations
        a = self.previous_action

        # get the updated epsilon, in case the slider was changed by the user
        self.epsilon = get_environment().epsilon

        # Update the Q value
        Q_old = self.predict(o, a)
        q = self.update(o, a, Q_old + self.alpha * (r - Q_old) )
        return True

class MyTilingRLAgent(MyTabularRLAgent):
    """
    Tiling RL Agent
    """
    def __init__(self, gamma, alpha, epsilon):
        """
        Constructor that is called from the robot XML file.
        Parameters:
        @param gamma reward discount factor (between 0 and 1)
        @param alpha learning rate (between 0 and 1)
        @param epsilon parameter for the epsilon-greedy policy (between 0 and 1)
        """
        MyTabularRLAgent.__init__(self, gamma, alpha, epsilon) # initialize the superclass
        self.O = {} # create a new dictionary to store observations at each (r, c)

    def get_possible_actions(self, observations):
        """
        Get the possible actions that can be taken given the state (observations)
        """
        rc_list = get_environment().maze.xy2rc(observations[0], observations[1])
        o = tuple(rc_list)

        aMin = self.action_info.min(0)
        aMax = self.action_info.max(0)
        actions = range(int(aMin), int(aMax+1))

        if o in self.O:
            for action in actions:
                if self.O[o][action] == 1:
                    actions.remove(action)

        #print "possible actions are: %s" % (actions)

        return actions

    def predict(self, observations, action):
        """
        Look up the Q-value for the given state (observations), action pair.
        """

        rc_list = get_environment().maze.xy2rc(observations[0], observations[1])
        o = tuple(rc_list)

        #print "predicting"
        #print "row: %s | col: %s | tuple: %s" % (row, col, o)

        if o not in self.Q:
            return 0
        else:
            return self.Q[o][action]

    def update(self, observations, action, new_value):
        """
        First, look at observations and record to self.O if there are any walls. Then, update the Q-function table with the new value for the (state, action) pair
        and update the blocks drawing.
        """

        #print "observations: %s, %s, %s, %s" % (observations[2],observations[3],observations[4],observations[5])

        actions = self.get_possible_actions(observations)

        rc_list = get_environment().maze.xy2rc(observations[0], observations[1])
        o = tuple(rc_list)

        if o not in self.O:
            self.O[o] = [0, 0, 0, 0]

        for x in range(0, 4):
            if observations[x+2]<1.0:
                self.O[o][x] = 1
                print "ELIMINATED ACTION: action %s at position %s, %s" % (x, o[0], o[1])

        #print "updating"
        #print "row: %s | col: %s | tuple: %s" % (row, col, o)

        if o not in self.Q:
            self.Q[o] = [0 for a in actions]
        self.Q[o][action] = new_value

        draw_o = tuple([x for x in observations])
        #print "o: %s" % (draw_o,)
        print "self.Q[o]: %s" % (self.Q[o])
        #print "action: %s" % (action)
        #print "new_value: %s" % (new_value)
        self.draw_q(draw_o)

    def draw_q(self, o):
        e = get_environment()
        if hasattr(e, 'draw_q'):
            rc_list = e.maze.xy2rc(o[0], o[1])
            tup_list = tuple(rc_list)
            q_values = self.Q[tup_list]

            draw_Q = {}
            min_row = rc_list[0]*20+10
            min_col = rc_list[1]*20+10

            for x in range(0, 8):
                for y in range(0, 8):
                    temp_list = (min_row, min_col, 1, 1, 1, 1)
                    temp_tuple = tuple(temp_list)
                    draw_Q[temp_tuple] = q_values
                    e.draw_q(temp_tuple, draw_Q)
                    #print "row: %s | col: %s" % (min_row, min_col)
                    min_row = min_row + 2.5
                min_col = min_col + 2.5
                min_row = min_row - 20

class MyNearestNeighborsRLAgent(MyTabularRLAgent):
    """
    Nearest Neighbors RL Agent
    """
    def __init__(self, gamma, alpha, epsilon):
        """
        Constructor that is called from the robot XML file.
        Parameters:
        @param gamma reward discount factor (between 0 and 1)
        @param alpha learning rate (between 0 and 1)
        @param epsilon parameter for the epsilon-greedy policy (between 0 and 1)
        """
        MyTabularRLAgent.__init__(self, gamma, alpha, epsilon) # initialize the superclass

        self.maze_walls = []
        for wall_set in get_environment().maze.walls:
            self.maze_walls.append(wall_set)
            inverted_set = (wall_set[1], wall_set[0])
            self.maze_walls.append(inverted_set)

        print "maze_walls: %s" % self.maze_walls
        self.weights = {}

    def predict(self, observations, action):
        """
        Look up the Q-value for the given state (observations), action pair.
        """
        x, y = observations[0], observations[1]

        #print "current position is %s, %s; we are considering action %s" % (x, y, action)

        #get new position
        if action==0:
            y = y + 2.5
        elif action==1:
            y = y - 2.5
        elif action==2:
            x = x + 2.5
        else:
            x = x - 2.5

        #print "after action, new position is %s, %s" % (x,y)

        r, c = get_environment().maze.xy2rc(x, y)

        all_neighbors = []
        for row in range(r-1, r+2):
            for col in range(c-1, c+2):
                if row >= 0 and row < ROWS and col >= 0 and col < COLS:
                    temp_list = (row,col)
                    all_neighbors.append(tuple(temp_list))
                else:
                    all_neighbors.append(0)

        not_neighbors = []

        for i in [1,3,5,7]:
            if all_neighbors[i] != 0 and (all_neighbors[i], all_neighbors[4]) in self.maze_walls:
                not_neighbors.append(i)

        if self.calculateCorners(all_neighbors, 0, 3, 1):
            not_neighbors.append(0)

        if self.calculateCorners(all_neighbors, 2, 1, 5):
            not_neighbors.append(2)

        if self.calculateCorners(all_neighbors, 6, 7, 3):
            not_neighbors.append(6)

        if self.calculateCorners(all_neighbors, 8, 7, 5):
            not_neighbors.append(8)

        for i in not_neighbors:
            all_neighbors[i] = 0

        reachable_neighbors = []

        for neighbor in all_neighbors:
            if neighbor != 0:
                reachable_neighbors.append(neighbor)

        #print "reachable neighbors of tile (%s, %s): %s" % (r, c, reachable_neighbors)

        min_distance1 = sys.maxint
        closest_neighbor1 = 0
        min_distance2 = sys.maxint
        closest_neighbor2 = 0
        min_distance3 = sys.maxint
        closest_neighbor3 = 0
        for neighbor in reachable_neighbors:
            neighbor_x = neighbor[0]*20+18.75
            neighbor_y = neighbor[1]*20+18.75
            diff_x = abs(observations[0] - neighbor_x)
            diff_y = abs(observations[1] - neighbor_y)
            distance = math.sqrt(diff_x*diff_x + diff_y*diff_y)
            if distance < min_distance1:
                min_distance3 = min_distance2
                closest_neighbor3 = closest_neighbor2
                min_distance2 = min_distance1
                closest_neighbor2 = closest_neighbor1
                min_distance1 = distance
                closest_neighbor1 = neighbor
            elif distance < min_distance2:
                min_distance3 = min_distance2
                closest_neighbor3 = closest_neighbor2
                min_distance2 = distance
                closest_neighbor2 = neighbor
            elif distance < min_distance3:
                min_distance3 = distance
                closest_neighbor3 = neighbor

            print "difference between current position: spot(%s, %s) at tile(%s, %s) and spot(%s, %s) at tile(%s, %s) is %s" % (observations[0], observations[1], r, c, neighbor_x, neighbor_y, neighbor[0], neighbor[1], distance)

        print "closest neighbors are %s ----- %s ----- %s" %(closest_neighbor1, closest_neighbor2, closest_neighbor3)

        print "self.Q is %s" % (self.Q)

        rclist = (r,c)
        rc = tuple(rclist)
        return self.findValue(closest_neighbor1, closest_neighbor2, closest_neighbor3, min_distance1, min_distance2, min_distance3)

    def calculateCorners(self, all_neighbors, x, y, z):
        if all_neighbors[x] != 0 and all_neighbors[y] != 0 and all_neighbors[z] != 0:
            return ((all_neighbors[x],all_neighbors[y]) in self.maze_walls or (all_neighbors[y],all_neighbors[4]) in self.maze_walls) and ((all_neighbors[x],all_neighbors[z]) in self.maze_walls or (all_neighbors[z],all_neighbors[4]) in self.maze_walls)
        return True

    def findValue(self, neighbor_a, neighbor_b, neighbor_c, dist_a, dist_b, dist_c):
        print "findValue called"
        dist_sum = dist_a + dist_b + dist_c
        weight_a = 1 - (dist_a/dist_sum)
        weight_b = 1 - (dist_b/dist_sum)
        weight_c = 1 - (dist_c/dist_sum)
        if neighbor_a not in self.Q:
            old_val_a = 0
        else:
            old_val_a = self.Q[neighbor_a]
        if neighbor_b not in self.Q:
            old_val_b = 0
        else:
            old_val_b = self.Q[neighbor_b]
        if neighbor_c not in self.Q:
            old_val_c = 0
        else:
            old_val_c = self.Q[neighbor_c]
        self.weights = {neighbor_a: weight_a, neighbor_b: weight_b, neighbor_c: weight_c}
        print "weights for %s, %s, %s are %s" % (neighbor_a, neighbor_b, neighbor_c, self.weights)
        return old_val_a * weight_a + old_val_b * weight_b + old_val_c * weight_c

    def update(self, observations, action, new_value):
        """
        Update the Q-function table with the new value for the (state, action) pair
        and update the blocks drawing.
        """
        print "update called"
        o = tuple([x for x in observations])
        actions = self.get_possible_actions(observations)
        if o not in self.Q:
            self.Q[o] = [0 for a in actions]
        self.Q[o][action] = new_value
        self.draw_q(o)

    def act(self, time, observations, reward):
        """
        return an action given the reward for the previous action and the new observations
        @param time current time
        @param observations a DoubleVector of observations for the agent (use len() and [])
        @param the reward for the agent
        """
        # get the reward from the previous action
        r = reward[0]

        # get the updated epsilon, in case the slider was changed by the user
        self.epsilon = get_environment().epsilon

        # get the old Q value
        Q_old = self.predict(self.previous_observations, self.previous_action)

        # get the max expected value for our possible actions
        (max_action, max_value) = self.get_max_action(observations)

        # update the Q values for neighboring tiles
        print "self.weights ---> %s" %self.weights
        for (neighbor, weight) in self.weights:
            if neighbor != 0:
                print "neighbor is ----------------> %s" % neighbor
                self.update( \
                    self.previous_observations, \
                    self.previous_action, \
                    self.Q[neighbor] + self.alpha * weight * (r + self.gamma * max_value - Q_old) )

        # select the action to take
        action = self.get_epsilon_greedy(observations, max_action, max_value)
        self.previous_observations = observations
        self.previous_action = action
        return action

    def draw_q(self, o):
        e = get_environment()
        if hasattr(e, 'draw_q'):
            rc_list = e.maze.xy2rc(o[0], o[1])
            tup_list = tuple(rc_list)
            q_values = self.Q[tup_list]

            draw_Q = {}
            min_row = rc_list[0]*20+10
            min_col = rc_list[1]*20+10

            for x in range(0, 8):
                for y in range(0, 8):
                    temp_list = (min_row, min_col, 1, 1, 1, 1)
                    temp_tuple = tuple(temp_list)
                    draw_Q[temp_tuple] = q_values
                    e.draw_q(temp_tuple, draw_Q)
                    #print "row: %s | col: %s" % (min_row, min_col)
                    min_row = min_row + 2.5
                min_col = min_col + 2.5
                min_row = min_row - 20
