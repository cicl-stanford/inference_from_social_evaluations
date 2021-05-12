import os
import numpy as np
import pandas as pd
import itertools
from scipy.stats import norm, entropy
from scipy.special import logsumexp
from action_blame import *

# loading prior distribution
# uniform over situations except 0 when the maximum possible reward is accomplished
prior = pd.read_csv('intermediate_analyses/prior.csv')

"""
compute reward outcomes and blame for all possible combination of situations and choices in those situations
parameters:
	n_trees			int		# trees that could block the way
	n_strengths		int		# strengths a fisher could have
	n_actions  		int		# actions fisher could take (trees or fish)
	n_people		int		# fishers
	blame_func		fn 		function to determine blame
return:
	situations_concat 		pd df   all possible situations and choices in them, as well as blame judgments
"""
def situations_df(n_trees, n_strengths, n_actions, n_people, w=.6, rationality_beta=1.5, k=2):
	# enumerating all possible combinations of number of trees, strengths, and actions for n people
	# 648 for standard possibilities of 3 trees, 3 strengths, 2 actions, and 3 people
	dd = pd.DataFrame(
		np.asarray(list(itertools.product(range(1, n_trees+1),
		np.asarray(list(itertools.product(range(1, n_strengths+1), repeat=n_people))),
		np.asarray(list(itertools.product(range(n_actions-1, -1, -1), repeat=n_people)))))))
	situations_enumerate = pd.concat((
		pd.DataFrame(dd[0]),
		pd.DataFrame(dd[1].to_list()),
		pd.DataFrame(dd[2].to_list())),
		axis=1)
	situations_enumerate.columns = ['trees', 'strength_a', 'strength_b', 'strength_c', 'action_a', 'action_b', 'action_c']

	# getting reward for each state
	rewards_store = []
	for situation_ix in range(situations_enumerate.shape[0]):
		situation = situations_enumerate.iloc[situation_ix]
		rewards_store.append(rewards(
			strengths= situation[['strength_a','strength_b','strength_c']].values, 
			trees=     situation[['trees']].values, 
			choices=   situation[['action_a','action_b','action_c']].values))
	situations_enumerate['reward'] = rewards_store

	# blame for each state
	blame_array = np.empty((situations_enumerate.shape[0], n_people))
	blame_array[:] = np.NaN
	for situation_ix in range(situations_enumerate.shape[0]):
		situation = situations_enumerate.iloc[situation_ix]

		# all possible rewards for different actions given trees and strengths 'situation'
		if situation_ix % 8 == 0:
			R_table = situations_enumerate.iloc[situation_ix:situation_ix+8][['action_a','action_b','action_c','reward']]

		# blame judgments for each agent
		for agent_ix in range(n_people):
			blame_array[situation_ix,agent_ix] = rationality_pivotality_opt(R_table=R_table, 
				k=k, 
				beta=rationality_beta, 
				choices=situation[['action_a','action_b','action_c']].values, 
				agent=agent_ix, 
				w=w)

	# combining blame with the situations df
	blame_array = pd.DataFrame(blame_array, columns=['blame_a','blame_b','blame_c'])
	situations_concat = pd.concat((situations_enumerate, blame_array), axis=1)

	return situations_concat

# computing the prior probability distribution
# p=0 when optimal is accomplished, uniform elsewhere
# pre-computed in intermediate_analyses/prior.csv
def compute_prior(n_trees=3, n_strengths=3, n_actions=2, n_people=3):
	prior = situations_df(n_trees, n_strengths, n_actions, n_people) 
	prior['max'] = prior.groupby(['trees','strength_a','strength_b','strength_c']).transform(max)['reward'] 
	prior['prior'] = 1/prior.shape[0] 
	prior.loc[prior['reward'] == prior['max'],'prior'] = 0
	prior['prior'] = prior['prior']/np.sum(prior['prior'])
	prior.to_csv('intermediate_analyses/prior.csv')

"""
Takes blame judgments and known situational factors and outputs a posterior over situations
parameters:
	df 			  df 			dataframe of situations and blame
	prior 		  df 			prior probability of states after accounting for 0 probability of maximum reward states
	other		  int			situation-specific parameters to condition on
	blame 		  float			blame judgments
	scale 		  float 		sd of comparison between given blames and model blames
	decision_beta float 		softmax temperature over options
return:
	df			  df 			posterior over blame
"""
def situation_inf(df,
	prior      = prior, 
	trees      = None, 
	strength_a = None, 
	strength_b = None, 
	strength_c = None, 
	action_a   = None, 
	action_b   = None, 
	action_c   = None, 
	reward     = None, 
	blame_a    = None, 
	blame_b    = None, 
	blame_c    = None,
	scale      = .1,
	decision_beta = 1):
	
	# setting prior probability
	df = df.copy()
	df['post'] = prior['prior']

	# removing conditioned-upon states
	variables = ['trees','strength_a','strength_b','strength_c','action_a','action_b','action_c']
	for variable in variables:
		if eval(variable) is not None:
			df = df[df[variable] == eval(variable)]

	# conditioning on reported blame
	blames = ['blame_a','blame_b','blame_c']
	for blame_idx in blames:
		if eval(blame_idx) is not None:
			tmp_lk = norm.pdf(eval(blame_idx), loc=df[blame_idx], scale=scale)
			df['post'] = df['post'] * tmp_lk
	df['post'] = df['post']/np.nansum(df['post'])

	# softmaxing by decision
	if decision_beta is not None:
		for variable in variables:
			if eval(variable) is None:
				tmp = df[[variable,'post']].groupby(variable).sum()
				tmp2 = np.exp(tmp*decision_beta)/np.sum(np.exp(tmp*decision_beta))
				tmp2 = tmp2.rename(columns = {'post':'post_' + variable})
				df = df.merge(tmp2, on=variable)	
		# finding all columns and multiplying to get softmaxed posterior
		post_names = df.filter(regex='^post_',axis=1)
		df['softmax'] = df.filter(post_names, axis=1).apply(np.prod, axis=1)
		df.drop(['post'], axis=1, inplace=True)
		df.drop(list(df.filter(regex = '^post_')), axis=1, inplace=True)
		df.rename(columns = {'softmax':'post'}, inplace=True)

	# correcting for underflow
	if any(np.isnan(df['post'])):
			df['post'] = 1e-10

	return df
	
