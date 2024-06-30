import numpy as np
import warnings

from .cube_utils import BatchCube, position_permutations, color_permutations, opp_action_permutations

action_count = 12
constant_priors = np.array([1/3] * action_count)
constant_value = .01
max_depth_value = 0.0

class State():
    """ 
    This is application specfic.
    This State object should be treated as immutable.
    We do not store the key, the input array, or a *copy* of the history since 
    that puts too much load on the memory (and the bottleneck in speed is the 
    neural network, not the tree search).
    """
    def __init__(self, history=1, random_depth=None, _internal_state=None):
        if _internal_state is not None:
            self._internal_state = _internal_state
        else:
            blank_history = tuple(None for _ in range(history-1))
            cube = BatchCube(1)
            if random_depth is not None:
                cube.randomize(random_depth)
            self._internal_state = (cube, ) + blank_history

    # no need for a copy since State is essentially immutable
    #def copy(self):
    #    return State(_internal_state = self.internal_state)

    def next(self, action):
        next_cube = self._internal_state[0].copy()
        next_cube.step(action)
        
        # to save memory, don't copy history 
        next_internal_state = (next_cube, ) + self._internal_state[:-1]

        return State(_internal_state=next_internal_state)

    def input_array(self):
        bit_arrays = []
        for c in self._internal_state:
            if c is None:
                bit_array = np.zeros((1, 54, 6), dtype=bool)
            else:
                bit_array = c.bit_array().reshape((1, 54, 6))
            bit_arrays.append(bit_array)
        
        return np.concatenate(bit_arrays, axis=0)
    
    def input_array_no_history(self):
        """
        Just return the newest state
        """
        bit_array = self._internal_state[0].bit_array().reshape((1, 54, 6))
        return bit_array 

    def key(self):
        return self.input_array().tobytes()

    def done(self):
        cube = self._internal_state[0]
        return cube.done()[0]

    def __str__(self):
        return str(self._internal_state)

class MCTSNode():
    def __init__(self, mcts_agent, state):
        self.state = state
        self.terminal = state.done()

        if not self.terminal:
            self.c_puct = mcts_agent.c_puct
            self.is_leaf_node = True
            self.prior_probabilities, self.node_value = mcts_agent.model_policy_value(state.input_array())
            self.total_visit_counts = 0
            self.visit_counts = np.zeros(action_count, dtype=int)
            self.total_action_values = np.zeros(action_count)
            self.mean_action_values = np.zeros(action_count)
            self.connected_to_terminal = np.zeros(action_count, dtype=bool)
            self.children = [None] * action_count

    def upper_confidence_bounds(self):
        return (self.node_value * self.c_puct * np.sqrt(self.total_visit_counts)) * self.prior_probabilities / (1 + self.visit_counts)

    def child(self, mcts_agent, action):
        # return node if already indexed
        child_node = self.children[action]
        if child_node is not None:
            return child_node
        
        # check transposition table
        next_state = self.state.next(action)
        key = next_state.key()
        if mcts_agent.transposition_table is not None and key in mcts_agent.transposition_table:
            node = mcts_agent.transposition_table[key]
            self.children[action] = node
            return node

        # create new node
        new_node = MCTSNode(mcts_agent, next_state)
        self.children[action] = new_node
        if mcts_agent.transposition_table is not None:
            mcts_agent.transposition_table[key] = new_node
        return new_node

    def select_leaf_and_update(self, mcts_agent, max_depth):
        #print("Entering Node:", self.state, "Depth:", max_depth)
        # terminal nodes are good
        if self.terminal:
            #print("... terminal node, returning 1")
            # record shortest distance to target
            depth = mcts_agent.max_depth - max_depth
            if depth < mcts_agent.shortest_path:
                mcts_agent.shortest_path = depth

            return 1., True

        # we stop at leaf nodes
        if self.is_leaf_node:
            self.is_leaf_node = False
            #print("... leaf node, returning ", self.node_value)
            return self.node_value, False

        # reaching max depth is bad
        # (this should punish loops as well)
        if not max_depth:
            #print("... max_depth == 0, returning -1")
            return max_depth_value, False

        # otherwise, find new action and follow path
        if self.total_visit_counts:
            action = np.argmax(self.mean_action_values + self.upper_confidence_bounds())
        else:
            action = np.argmax(self.prior_probabilities) # use prior on first move since mean_action_values and upper_confidence_bounds are all zero
        
        # update visit counts before recursing in case we come across the same node again
        self.total_visit_counts += 1
        self.visit_counts[action] += 1

        #print("Performing Action:", action)
        child_action_value, reached_terminal = \
            self.child(mcts_agent, action).select_leaf_and_update(mcts_agent, max_depth - 1)
        action_value = mcts_agent.gamma * child_action_value

        #print("Returning back to:", max_depth)

        # record if reached_terminal
        if reached_terminal:
            self.connected_to_terminal[action] = True

        # recursively update edge values
        self.total_action_values[action] += action_value
        self.mean_action_values[action] = self.total_action_values[action] / self.visit_counts[action]

        # recusively backup leaf value
        #print("DB: update node", "action:", action, "action value:", action_value)
        #print(self.status() + "\n")

        return action_value, reached_terminal

    def action_visit_counts(self):
        """ Returns action visit counts. """
        if self.terminal or self.is_leaf_node:
            return np.zeros(action_count, dtype=int)

        return self.visit_counts

    def action_probabilities(self, inv_temp):
        # if no exploring, then this is not defined
        if self.terminal or self.is_leaf_node:
            return None

        if inv_temp == 1:
            return self.visit_counts / self.visit_counts.sum()
        else:
            # scale before exponentiation (the result is the same, but less likely to overflow)
            exponentiated_visit_counts = (self.visit_counts / self.visit_counts.sum()) ** inv_temp
            return exponentiated_visit_counts / exponentiated_visit_counts.sum() 

    def status(self):
        stats = "Node Status:\n"
        stats += str(self.state) + "\n"

        if self.terminal:
            stats += "terminal node"
            return stats

        stats += "Leaf: " + str(self.is_leaf_node) + "\n"
        stats += "Value: " + str(self.node_value) + "\n"
        stats += "Priors: " + str(self.prior_probabilities) + "\n"
        stats += "PUCT: " + str(self.upper_confidence_bounds()) + "\n"
        stats += "Visit Cnts: " + str(self.visit_counts) + "\n"
        stats += "Total Action Values: " + str(self.total_action_values) + "\n"
        stats += "Mean Action Values: " + str(self.mean_action_values)

        return stats

class MCTSAgent():
    def __init__(self, model_policy_value, initial_state, max_depth, transposition_table={}, c_puct=1.0, gamma=.95, dirichlet_const=1/12):
        self.model_policy_value = model_policy_value
        self.max_depth = max_depth
        self.total_steps = 0
        self.transposition_table = transposition_table
        self.c_puct = c_puct  # exploration constant
        self.gamma = gamma  # decay constant
        self.dirichlet_const = dirichlet_const # alpha (None if no Dirichlet noise)

        self.initial_node = MCTSNode(self, initial_state)
        if self.dirichlet_const is None:
            self.initial_node.prior_probabilities = self.model_policy_value(self.initial_node.state.input_array())[0]
        else:    
            self.initial_node.prior_probabilities = \
                .75 * self.model_policy_value(self.initial_node.state.input_array())[0] +\
                .25 * np.random.dirichlet([self.dirichlet_const]*action_count, 1)[0]

        self.shortest_path = self.max_depth + 1

    def search(self, steps: int, stop_early: bool = False):
        self.initial_node.is_leaf_node = False # so that at least exactly one move if steps = 1
        for s in range(steps):
            if stop_early and self.shortest_path < self.max_depth + 1:
                break
            self.initial_node.select_leaf_and_update(self, self.max_depth) # explore new leaf node
            self.total_steps += 1

    def action_visit_counts(self):
        return self.initial_node.action_visit_counts()
    
    def action_probabilities(self, inv_temp):
        return self.initial_node.action_probabilities(inv_temp)

    def initial_node_status(self):
        return self.initial_node.status()

    def is_terminal(self):
        return self.initial_node.terminal

    def advance_to_best_child(self):
        """ Advance to the best child node """
        
        best_action = np.argmax(self.action_visit_counts())
        self.advance_to_action(best_action)

    def advance_to_action(self, action):
        """ Advance to a child node via the given action """
        
        # TOFIX: I should (maybe?) find a way to delete the nodes not below this one, 
        # including from the tranposition table
        
        self.initial_node = self.initial_node.child(self, action) 
        if self.dirichlet_const is None:
            self.initial_node.prior_probabilities = self.model_policy_value(self.initial_node.state.input_array())[0]
        else:    
            self.initial_node.prior_probabilities = \
                .75 * self.model_policy_value(self.initial_node.state.input_array())[0] +\
                .25 * np.random.dirichlet([self.dirichlet_const]*action_count, 1)[0]
        
        self.shortest_path = self.max_depth + 1

    def stats(self, key):
        """ Proviods various stats on the MCTS """
        
        if key == 'shortest_path':
            return self.shortest_path if self.shortest_path <= self.max_depth else -1
        elif key == 'prior':
            return self.model_policy_value(self.initial_node.state.input_array())[0]
        elif key == 'prior_dirichlet':
            return self.initial_node.prior_probabilities
        elif key == 'value':
            return self.model_policy_value(self.initial_node.state.input_array())[1]
        elif key == 'visit_counts':
            return self.initial_node.visit_counts
        elif key == 'total_action_values':
            return self.initial_node.total_action_values
        else:
            warnings.warn("'{}' argument not implemented for stats".format(key), stacklevel=2)
            return None
