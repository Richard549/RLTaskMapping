import numpy as np
import copy
from agent import *

class Transition():

	def __init__(self,
		pre_state,
		action,
		post_state,
		reward,
		termination,
		actionable_nodes_at_post_state=None,
		actionable_nodes_at_pre_state=None,
		action_probability=None,
		allocated_tasks=None,
		rnd_target_output=None,
		pre_state_value=None,
		post_state_value=None,
		intrinsic_reward=None):

		self.pre_state = State(np.copy(pre_state.feature_matrix),partial_solution_node_indexes=np.copy(pre_state.partial_solution_node_indexes),unselected_independent_node_indexes=np.copy(pre_state.unselected_independent_node_indexes))
		self.post_state = State(np.copy(post_state.feature_matrix),partial_solution_node_indexes=np.copy(post_state.partial_solution_node_indexes),unselected_independent_node_indexes=np.copy(post_state.unselected_independent_node_indexes))

		self.action = Action(action.node_idx,action.label)
		self.reward = reward
		self.termination = termination

		if actionable_nodes_at_post_state is not None:
			self.actionable_nodes_at_post_state = np.copy(actionable_nodes_at_post_state)
		if actionable_nodes_at_pre_state is not None:
			self.actionable_nodes_at_pre_state = np.copy(actionable_nodes_at_pre_state)
		if action_probability is not None:
			self.action_probability = action_probability
		if allocated_tasks is not None:
			self.allocated_tasks = allocated_tasks # if the transition is to be executed, it should have some associated allocated_tasks
		if rnd_target_output is not None:
			self.rnd_target_output = rnd_target_output 
		if pre_state_value is not None:
			self.pre_state_value = pre_state_value 
		if post_state_value is not None:
			self.post_state_value = post_state_value 
		if intrinsic_reward is not None:
			self.intrinsic_reward = intrinsic_reward
			self.normalized_intrinsic_reward = 0

class ReplayMemory():

	def __init__(self, max_size):
		self.memory = []
		self.max_size = max_size

	def add(self, transition):

		if len(self.memory) == self.max_size:
			self.memory.pop(0)

		self.memory.append(transition)

	def sample_minibatch(self, minibatch_size):
		minibatch = np.random.choice(self.memory, minibatch_size, replace=False)

		return minibatch

	def get_ordered_transitions(self):
		return self.memory

	def reset(self):
		self.memory.clear()


		
