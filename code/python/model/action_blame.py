import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import itertools


# choice = 0 - fish
# choice = 1 - trees

"""
Get reward of actual event
parameters:
	strengths 		list 			strength of each fisherman
	trees 			int 			number of trees
return:
	R 				np.array 		total reward for each trial
"""
def rewards(strengths, trees, choices):
	# number of fishermen
	n = len(strengths) 

	# indices of who is getting the trees
	tree_guys = [j for j, x in enumerate(choices) if x] 

	# indices of those who are not getting trees (getting fish)
	fish_guys = list(set(range(n)) - set(tree_guys))

	# total strength of guys going for trees
	strength_sum = np.sum([strengths[tree_guy] for tree_guy in tree_guys]) 

	# if strength_sum is larger than trees, they cleared the trees
	if strength_sum >= trees:
		# R is sum of rewards 
		R = np.sum([strengths[fish_guy] for fish_guy in fish_guys]) 
	else:
		R = 0

	return R

"""
Get reward table (array) of multiple trials
parameters:
	strengths 		list 			strength of each fisherman
	trees 			int 			number of trees
return:
	R 				np.array 		total reward for each trial
"""
def rewards_multi(strengths, trees):
	# number of fishermen
	n = len(strengths) 

	# initialize rewards table
	R = np.zeros(np.power(2,n))

	# set up truth table of all combinations of outcomes
	truth_table = np.asarray(list(itertools.product([True, False], repeat=n)))

	# for every combination in truth_table:
	for i in range(0,np.power(2,n)):
		# indices of who is getting the trees
		tree_guys = [j for j, x in enumerate(truth_table[i,:]) if x] 

		# indices of those who are not getting trees (getting fish)
		fish_guys = list(set(range(n)) - set(tree_guys))
		
		# total strength of guys going for trees
		strength_sum = np.sum([strengths[tree_guy] for tree_guy in tree_guys]) 

		# if strength_sum is larger than trees, they cleared the trees
		if strength_sum >= trees:
			# R[i] is sum of rewards for that trial (sum of fish caught)
			R[i] = np.sum([strengths[fish_guy] for fish_guy in fish_guys]) 
 
	return R

"""
Get expected reward table
parameters:
	R 				np.array 		rewards table
	probs 			np.array		probability of clearing trees? (choosing "0")
return:
	RP 				np.array		expected reward table
"""
def expected_rewards_multi_ZD(R_table, probs):
	# number of rows (trials)
	n = len(probs)

	# making a copy of R_table
	df = R_table.copy()
	# probabilities of different situations
	prob_mat = R_table.copy()

	# for each agent, replace truth table choice with the probability that they made that choice
	# for example, if agent 1's probability of clearing trees is .9, put that value in situations where he cleared
	for agent in range(n):
		prob_mat.iloc[:,agent] = prob_mat.iloc[:,agent].map({1: probs[agent], 0: 1-probs[agent]})

	# getting overall probability of each situation by multiplying probability of each action
	df['prob'] = prob_mat.iloc[:,range(n)].apply(np.prod,axis=1)

	# multiply probability of situation by reward in that situation
	df['EV'] = df['reward'] * df['prob']

	return df


def expected_rewards_multi(R, probs, **attr):
	# intitialize expected reward table
	RP = np.zeros(R.shape)

	# number of rows (trials)
	n = probs.shape[0]

	if 'truth_table' in attr:
		truth_table = attr['truth_table']
		
	# set up truth table of all combinations of outcomes
	else: 
		truth_table = np.asarray(list(itertools.product([True, False], repeat=n)))

	prob_row = []

	for i in range(len(truth_table)):
		# indices of who is getting the trees
		tree_guys = [j for j, x in enumerate(truth_table[i,:]) if x] 

		# indices of those who are not getting trees (getting fish)
		fish_guys = list(set(range(n)) - set(tree_guys))
		
		# initialize probability table for p(tree_guy gets tree) for each tree guy
		prob_row.append([probs[j] for j in tree_guys])
		
		# add probabilities that fish guys get trees to probability table
		prob_row[i] += ([(1-probs[j]) for j in fish_guys])

	prob_row = [np.product(t) for t in prob_row]
	
	normed = []
	for i in range(len(prob_row)):
		normed.append([])
		normed[i].append(prob_row[i]/np.sum(prob_row))

	# product of all elements (including reward), reward by the probability of clearing the trees
	for i in range(0,len(truth_table)):
		normed[i].append(R[i])

	for i in range(len(normed)):
		RP[i] = np.product(normed[i])

	return RP

"""
vector of utilities
parameters: 
	u 				float	 		expected utility
	beta 			float 			rationality parameter
return:
	rationality 	np.array 		soft max probability agent should take action
"""
def logit(u, beta):
	# print 'logit', u
	return np.exp(beta * u) / np.sum(np.exp(beta * u))

"""
get list of probability that an agent should clear trees
parameters:
	R 				np.array 		rewards table
	k 				int				
	beta 			float
return:
	probs 			list 			[should fish, should clear trees]	
"""
def probability_k_zd(R_table, k, beta):
	# if rewards table isn't empty
	if R_table.shape[0] != 0:
		# get log 2 (number of rows in R?)
		# 2**n = len(R)
		n = int(np.log2(R_table.shape[0]))

		# initialize probability table to .5 for all n
		probs = 0.5*np.ones((n))

		# set utils to be array([0., 0.])
		utils = np.zeros(2)

		# for k depth
		for i in range(int(k)):

			# get expected rewards table
			RP = expected_rewards_multi_ZD(R_table, probs)

			# dealing with extreme cases
			probs[probs==1] = 1-1e-12
			probs[probs==0] = 1e-12

			# for each agent
			for agent in range(n):
				# expected values for trials where the agent cleared trees
				utils[0] = np.sum(RP[RP.iloc[:,agent]==1][['EV']])

				# expected values for trials where the agent went fishing
				utils[1] = np.sum(RP[RP.iloc[:,agent]==0][['EV']])

				# assign probability of the agent clearing trees to each agent
				probs[agent] = logit(utils,beta)[0]
 
		return probs #probability that fishermen should clear trees
	else:
		return 1

def probability_k(R, k, beta):
	# if rewards table isn't empty
	if R.size != 0:
		# get log 2 (number of rows in R?)
		# 2**n = len(R)
		n = int(np.log2(len(R)))

		# initialize probability table to .5 for all n
		probs = 0.5*np.ones((n))

		# get truth table
		truth_table = np.asarray(list(itertools.product([True, False], repeat=n)))

		# set utils to be array([0., 0.])
		utils = np.zeros(2)

		# for k depth
		for i in range(int(k)):

			# get expected rewards table
			RP = expected_rewards_multi(R, probs)
			# for each agent
			for agent in range(n):
				# compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
				# list of indices such that the agent chose to cleared trees?
				indices = list(itertools.compress(range(len(truth_table[:,agent])), truth_table[:,agent]))

				########### ZD NOTE: SWITCHED AROUND WHICH UTILS, I THINK THIS WAS WRONG
				# dealing with extreme cases
				if probs[agent] == 0.0:
					probs[agent] = 1e-12
				if probs[agent] == 1.0:
					probs[agent] = 1-1e-12

				# ORIGINAL
				# 1 over probability the agent cleared trees * sum of expected rewards for all trials that agent cleared trees
				utils[0] = 1./probs[agent]*np.sum([RP[index] for index in indices])
				# 1 over the probability the agent went fishing * sum of expected rewards for all trials that agent did not clear trees
				utils[1] = 1./(1-probs[agent])*np.sum([RP[index] for index in list(set(range(len(R)))-set(indices))])

				# assign probability of the agent clearing trees to 
				probs[agent] = logit(utils,beta)[1]
 
		return probs #probability that fishermen should clear trees
	else:
		return 1
"""
get list of list of probability that an agent should clear trees for each k depth
parameters:
	R 				np.array 		rewards table
	k 				int				
	beta 			float
return:
	probs 			list 			[should fish, should clear trees]	
"""
def multi_probability_k(R, k, beta):
	# if rewards table isn't empty
	if R.size != 0:
		# get log 2 (number of rows in R?)
		# 2**n = len(R)
		n = int(np.log2(len(R)))

		# initialize probability table to .5 for all n
		probs = [0.5*np.ones((n)) for i in range(k+1)]

		# get truth table
		truth_table = np.asarray(list(itertools.product([True, False], repeat=n)))

		# set utils to be array([0., 0.])
		utils = np.zeros(2)
		
		# for k depth
		for i in range(k):

			# get expected rewards table
			RP = expected_rewards_multi(R, probs[i])

			# for each agent
			for agent in range(0,n):
				# compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
				# list of indices such that the agent chose to cleared trees?
				indices = list(itertools.compress(range(len(truth_table[:,agent])), truth_table[:,agent]))

				# 1 over the probability the agent went fishing * sum of expected rewards for all trials that agent did not clear trees
				if probs[i][agent] != 1.0:
					utils[0] = 1./(1-probs[i][agent])*np.sum([RP[index] for index in list(set(range(len(R)))-set(indices))])
				else:
					probs[i][agent] = 1-1e-12
					utils[0] = 1./(1-probs[i][agent])*np.sum([RP[index] for index in list(set(range(len(R)))-set(indices))])
				if probs[i][agent] != 0.0:
					# 1 over probability the agent cleared trees * sum of expected rewards for all trials that agent cleared trees
					utils[1] = 1./probs[i][agent]*np.sum([RP[index] for index in indices])
				else:
					probs[i][agent] = 1e-12
					utils[1] = 1./probs[i][agent]*np.sum([RP[index] for index in indices])
				
				# assign probability of the agent clearing trees to 
				probs[i+1][agent] = logit(utils,beta)[1]
			
		return probs[1:] #probability that fishermen should clear trees
	else:
		return 1


"""
get pivotality to any reward
parameters:
	R 				np.array 		rewards table
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
return:
	pivotality 		float 			pivotality to optimal rewards
"""
def pivotality_any(R, choices, agent):
	# number of agents 
	n = len(choices)

	# set up truth values for ever possible combo
	truth_table = np.asarray(list(itertools.product([1, 0], repeat=n)))
	
	#checking for cases where it's impossible to get any reward
	if np.sum(R.flatten()) == 0: 
		best_choices = np.array([[1,1,1]])
		other_agents = None
	else:
		# get indices of R with optimal reward
		best_indices = np.argwhere(R)
		# array of the rows in the truth table that produce an optimal reward
		best_choices = np.asarray([truth_table[index].flatten() for index in best_indices])

		other_agents = list(set(range(len(best_choices[0]))) - set([agent]))

	# if, for all possible positive solutions or agent made a good choice, the agent made a good choice, they are not pivotal
	if all([best_choice == choices[agent] for best_choice in best_choices[:,agent]]) or any(best_choice == choices for best_choice in best_choices):
		return 0.
	# elif, they reveived a reward and without they they would not have, they are pivotal
	elif len([c for c in best_choices if all(np.all(c[a] == choices[a] for a in other_agents))]) > 1:
		return 1.
	# else, 
	else:
		# gets all the changes for each of the optimal scenarios
		# changes listed in columns, changes in row 1 -> col 1
		all_changes = np.transpose([abs(best_choice-np.array(choices)) for best_choice in best_choices]) 

		# count changes (sums each column)
		changes = sum(all_changes)

		# finds indices where closest optimal choices
		min_best = np.argwhere(abs(changes - int(np.amin(changes))) < 0.000001)

		if all(best_choices[min_best.flatten()][:,agent] == choices[agent]): 
			# if closest optimal world has same choice as agent, no pivotality 
			return 0
		elif len(min_best.flatten()) > 1 and not all(best_choices[min_best.flatten()][:,agent] == 1 - choices[agent]) and choices[agent] == 1: 
			# handling edge cases
			index_use = np.argwhere(best_choices[:,agent] == choices[agent])
			return 1./(changes[index_use].flatten()[0]+1)
		else:
			return 1./np.min(changes)

"""
get pivotality to desired reward (optimal or any reward)
parameters:
	R 				np.array 		rewards table
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
	opt 			boolean 		True: distance to optimal reward; False: distance to any reward
return:
	pivotality 		float 			pivotality to optimal rewards
"""
def pivotality_zd(R_table, choices, agent, opt=True):
	# if it's not possible to get reward, blame is 0
	if np.sum(R_table['reward']) == 0: 
		return 0

	if opt:
		# finding the rows in the truth table that produce an *optimal* reward
		best_choices = R_table[R_table['reward'].values == R_table['reward'].values.max()]
	else:
		# finding the rows in the truth table that produce *any* reward
		best_choices = R_table[R_table['reward'].values > 0]
		
	# compute the changes to choices that would need to be made to get to the desired reward outcome
	n_agents = len(choices)
	changes = best_choices.iloc[:,:n_agents].apply(lambda x: x - choices, axis=1)

	# only consider situations where the reward outcome was counterfactually dependent on the agent
	changes = changes[changes.iloc[:,agent] != 0]
	if changes.shape[0] == 0:
		# if the agent always made the right choice, pivotality is 0
		return 0

	# pivotality is the number of changes that would need to be made to others to make one's own actions pivotal
	other_changes = changes.drop(changes.columns[agent], axis=1) 
	changes['other_changes'] = other_changes.apply(lambda x: sum(abs(x)), axis=1)

	# 1/(N+1), where N is changes that other agents would need to make for agent's actions to be pivotal
	changes['pivotality'] = 1/(changes['other_changes'] + 1) 

	# max because 1/(minimum number of changes) := maximum pivotality
	return np.max(changes['pivotality'])

def pivotality_opt(R, choices, agent):
	# number of agents 
	n = len(choices)

	# set up truth values for ever possible combo
	truth_table = np.asarray(list(itertools.product([1, 0], repeat=n)))
	
	#checking for cases where it's impossible to get any reward
	if np.sum(R.flatten()) == 0: 
		best_choices = np.array([[1,1,1]])
	else:
		# get indices of R with optimal reward
		best_indices = np.argwhere(R == np.amax(R))
		# array of the rows in the truth table that produce an optimal reward
		best_choices = np.asarray([truth_table[index].flatten() for index in best_indices])
		
	# if, for all possible optimal solutions, the agent made the optimal choice, they are not pivotal
	if all([best_choice == choices[agent] for best_choice in best_choices[:,agent]]):
		return 0
	else:
		# gets all the changes for each of the optimal scenarios
		# changes listed in columns, changes in row 1 -> col 1
		all_changes = np.transpose([abs(best_choice-np.array(choices)) for best_choice in best_choices]) 

		# count changes (sums each column)
		changes = sum(all_changes)

		# finds indices where closest optimal choices
		min_best = np.argwhere(abs(changes - int(np.amin(changes))) < 1e-6)

		if all(best_choices[min_best.flatten()][:,agent] == choices[agent]): 
			# if closest optimal world has same choice as agent, no pivotality 
			return 0
		elif len(min_best.flatten()) > 1 and not all(best_choices[min_best.flatten()][:,agent] == 1 - choices[agent]) and choices[agent] == 1: 
			# handling edge cases
			index_use = np.argwhere(best_choices[:,agent] == choices[agent])
			return 1./(changes[index_use].flatten()[0]+1)
		else:
			return 1./np.min(changes)

"""
get rationality
parameters:
	R 				np.array 		rewards table
	k 				
	beta 			float 			Beta, rationality parameter
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
return:
	rationality 	float 			rationality
"""
def rationality(R, k, beta, choices, agent):
	probs = probability_k(R, k, beta)
	if choices[agent] == 1:
		# if agent chose "1" (cleared trees?) 
		return 1-probs[agent]
	else:
		# if agent did not clear trees?
		return probs[agent]


"""
get weight sum of agent's pivotality to optimal and probability of action
parameters:
	R_table 		np.array 		rewards table
	k 				
	beta 			float 			Beta, rationality parameter
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
	w 				float 			weight
return:
	responsibility	float 			weighted sum of pivotality and probability of action
"""
def rationality_pivotality_opt(R_table, k, beta, choices, agent, w):
	# probability that agent clears trees
	#p_trees = probability_k(R,k,beta)[agent]
	p_trees = probability_k_zd(R_table,k,beta)[agent]
	
	# pivotality of agent
	#pivotality = pivotality_opt(R,choices,agent)
	pivotality = pivotality_zd(R_table,choices,agent,opt=True)

	if choices[agent] == 1:
		# if agent went fishing
		# return (1 - probability agent cleared trees)*weight + (1 - weight)*pivotality
		return (1-p_trees)*w + (1-w)*pivotality
	else:
		# if agent cleared trees
		# return (probability agent cleared trees)*weight + (1 - weight)*pivotality
		return p_trees*w + (1-w)*pivotality

"""
qualitative version of pivotality plus rationality
"""
def rationality_expected(probs, strengths, trees, choices, agent, w):
		
	reward = rewards(strengths, trees, choices)

	counterfactual = list(choices)
	counterfactual[agent] = not choices[agent]

	
	if choices[agent] == 0:
		# if worse
		if reward > rewards(strengths, trees, counterfactual):
			return w*(1-probs) + (1-w)*(1)
		# if better
		elif reward < rewards(strengths, trees, counterfactual):
			return w*(1-probs) + (1-w)*(-1)
		# if no difference
		else:
			return w*(1-probs) + (1-w)*(0)
	else:
		# if worse
		if reward > rewards(strengths, trees, counterfactual):
			return w*(probs) + (1-w)*(1)
		# if better
		elif reward < rewards(strengths, trees, counterfactual):
			return w*(probs) + (1-w)*(-1)
		# if no difference
		else:
			return w*(probs) + (1-w)*(0)

"""
get weight sum of agent's pivotality to optimal and (optimal reward - actual reward) * probability of action
parameters:
	R 				np.array 		rewards table
	k 				int 			depth of recursion for rationality
	beta 			float 			Beta, rationality parameter
	strengths 		list 			strength of each fisherman
	trees 			int 			number of trees
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
	w 				float 			weight
return:
	responsibility	float 			weighted sum of pivotality and probability of action
"""
def rationality_reward_pivotality_opt(R_table, k, beta, strengths, trees, choices, agent, w):
	
	# probability that agent clears trees
	p_trees = probability_k_zd(R_table,k,beta)[agent]
	#p_trees = probability_k(R,k,beta)[agent]
	
	# pivotality of agent
	#R = R_table[['reward']].values.flatten()
	pivotality = pivotality_zd(R_table,choices,agent,opt=True)

	# optimal reward
	reward_opt = np.amax(R_table['reward'])

	# actual reward
	reward_act = rewards(strengths, trees, choices)


	# ZD: Not sure why weighting pivotality by change in rewards??
	if choices[agent] == 0:
		# return (1 - probability agent cleared trees)*weight + (1 - weight)*change in rewards*pivotality
		return w*(1-p_trees) + (1-w)*((reward_opt-reward_act)/reward_opt)*pivotality
	else:
		# return (probability agent cleared trees)*weight + (1 - weight)*change in rewards*pivotality
		return w*p_trees     + (1-w)*((reward_opt-reward_act)/reward_opt)*pivotality


def reward_exp(R, k, beta):
	probs = probability_k(R,k,beta)
	return sum(expected_rewards_multi(R, probs))


def exp_reward_action(R, probs, choices, agent):
	
	# index for exp reward if choose original choices. product of all prob of choosing those actions * reward if they do
	n = len(choices) 
	truth_table = list(itertools.product([True, False], repeat=n))

	# turn choices into boolian array
	choices = np.array(choices) > 0

	actual_table = [row for row in truth_table if row[agent] == choices[agent]]
	actual_rewards = [reward for row, reward in zip(truth_table, R) if row[agent] == choices[agent]]

	return sum(expected_rewards_multi(np.asarray(actual_rewards), probs, truth_table=np.asarray(actual_table)))

def exp_rewards_cond(R, s, probs, choices, agent):

	# index for exp reward if choose original choices. product of all prob of choosing those actions * reward if they do
	n = len(choices) 
	truth_table = list(itertools.product([True, False], repeat=n))

	# turn choices into boolian array
	choices = np.array(choices) > 0

	#initialize probability list for fishermen
	probs_conditioned = range(n)


	for fisherman in range(n):
		if fisherman == agent:
			probs_conditioned[agent] = not choices[agent]
		else:
			probs_conditioned[fisherman] = s*choices[fisherman] + (1-s)*probs[fisherman]

	
	counterfactual_table = [row for row in truth_table if row[agent] == probs_conditioned[agent]]

	counterfactual_rewards = [reward for row, reward in zip(truth_table, R) if row[agent] == probs_conditioned[agent]]

	exp_rewards_conditioned = expected_rewards_multi(np.asarray(counterfactual_rewards), np.asarray(probs_conditioned), truth_table=np.asarray(counterfactual_table))


	return sum(exp_rewards_conditioned)

"""
expected reward given their action - reward given observation and if agent changes their choice
parameters:
	R 				np.array 		rewards table
	k 				int 			k depth
	beta 			float 			Beta, rationality parameter
	strengths 		list 			strength of each fisherman
	trees 			int 			number of trees
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
	w 				float 			weight
return:
	responsibility	float 			weighted sum of pivotality and probability of action
"""
def kemp_model_reward_exp_conditioned(R, k, beta, strengths, trees, choices, agent, s):
	
	# probability that agent clears trees
	probs = probability_k(R,k,beta)

	return exp_reward_action(R, probs, choices, agent) - exp_rewards_cond(R, s, probs, choices, agent)

def multi_kemp_model_reward_exp_conditioned(R, probs, strengths, trees, choices, agent, s):
	
	# probability that agent clears trees

	return exp_reward_action(R, probs, choices, agent) - exp_rewards_cond(R, s, probs, choices, agent)

def multi_kemp_model_reward_exp_conditioned_opt(R, probs, strengths, trees, choices, agent, s):
		
	# probability that agent clears trees

	return (exp_reward_action(R, probs, choices, agent) - exp_rewards_cond(R, s, probs, choices, agent)) / 	np.amax(R)


"""
actual reward - reward given observation and if agent changes their choice
parameters:
	R 				np.array 		rewards table
	k 				int 			k depth
	beta 			float 			Beta, rationality parameter
	strengths 		list 			strength of each fisherman
	trees 			int 			number of trees
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
	w 				float 			weight
return:
	responsibility	float 			weighted sum of pivotality and probability of action
"""
def kemp_model_reward_act(R, k, beta, s, strengths, trees, choices, agent):
	
	# probability that agent clears trees
	probs = probability_k(R,k,beta)

	return 	rewards(strengths, trees, choices) - exp_rewards_cond(R, s, probs, choices, agent)

def multi_kemp_model_reward_act(R, s, probs, strengths, trees, choices, agent):
		
	# probability that agent clears trees

	return (rewards(strengths, trees, choices) - exp_rewards_cond(R, s, probs, choices, agent)) 


def multi_kemp_model_reward_act_opt(R, s, probs, strengths, trees, choices, agent):
		
	# probability that agent clears trees

	return (rewards(strengths, trees, choices) - exp_rewards_cond(R, s, probs, choices, agent)) / 	np.amax(R)


"""
expected reward - reward given observation and if agent changes their choice
parameters:
	R 				np.array 		rewards table
	k 				int 			k depth
	beta 			float 			Beta, rationality parameter
	strengths 		list 			strength of each fisherman
	trees 			int 			number of trees
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
	w 				float 			weight
return:
	responsibility	float 			weighted sum of pivotality and probability of action
"""
def kemp_model_reward_exp(R, k, beta, strengths, trees, choices, agent, w, s):
	
	# probability that agent clears trees
	probs = probability_k(R,k,beta)


	return sum(expected_rewards_multi(R, probs)) - exp_rewards_cond(R, s, probs, choices, agent)


def multi_kemp_model_reward_exp(R, s, probs, strengths, trees, choices, agent):
		
	# probability that agent clears trees

	return (sum(expected_rewards_multi(R, probs)) - exp_rewards_cond(R, s, probs, choices, agent)) 


def multi_kemp_model_reward_exp_opt(R, s, probs, strengths, trees, choices, agent):
		
	# probability that agent clears trees

	return (sum(expected_rewards_multi(R, probs)) - exp_rewards_cond(R, s, probs, choices, agent)) / np.amax(R)

"""
expected reward - reward given observation and prob agent changes their choice
agents alternative choices are calculated with kemp as well
parameters:
	R 				np.array 		rewards table
	k 				int 			k depth
	beta 			float 			Beta, rationality parameter
	strengths 		list 			strength of each fisherman
	trees 			int 			number of trees
	choices 		list 			choices each agent made
	agent 			int 			index of agent we're evaluating
	w 				float 			weight
return:
	responsibility	float 			weighted sum of pivotality and probability of action
"""
def kemp_model(R, k, beta, strengths, trees, choices, agent, w, s):
	
	# probability that agent clears trees
	probs = probability_k(R,k,beta)

	return sum(expected_rewards_multi(R, probs)) - exp_rewards_cond(R, s, probs, choices, agent)

def norm(prediction):
	if prediction == 0:
		return prediction

	return prediction/30

def norm_reward_opt(R, prediction):
	if prediction == 0:
		return prediction

	return prediction/np.amax(R)

def norm_reward_exp(R, k, beta, prediction):

	probs = probability_k(R,k,beta)

	if prediction == 0:
		return prediction

	return prediction/sum(expected_rewards_multi(R, probs))