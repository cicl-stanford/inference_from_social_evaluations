import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import softmax

# reading in participant judgments and processing ------------------------
ppt_df = pd.read_csv('../../../data/cogsci2021/df.long.csv')
# format that the modeling code understands
ppt_df.loc[ppt_df['choice']=='fish', 'choice']  = 0
ppt_df.loc[ppt_df['choice']=='trees', 'choice'] = 1
ppt_df.choice = pd.to_numeric(ppt_df.choice) 
ppt_df = ppt_df.rename(columns = {'A_choice':'a_choice', 'B_choice':'b_choice', 'C_choice':'c_choice', 'A_strength':'a_strength', 'B_strength':'b_strength', 'C_strength':'c_strength'})
# cached modeling code
models_blame = pd.read_csv('./intermediate_analyses/models_blame.csv')
prior = pd.read_csv('./intermediate_analyses/prior.csv')

# list of cases ------------------------
cases = [
	{'id':0, 'trees':0, 'strengths':[1,1,3], 'actions':['fish','fish','fish'],    'blames':['high','high','low'],        'unknowns':['trees']},           # implies n_trees=1
	{'id':1, 'trees':0, 'strengths':[3,1,1], 'actions':['fish','fish','fish'],    'blames':['low','medium','medium'],    'unknowns':['trees']},       # implies n_trees=2
	{'id':2, 'trees':0, 'strengths':[1,3,1], 'actions':['fish','fish','fish'],    'blames':['low','high','low'],         'unknowns':['trees']},		    # implies n_trees=3
	{'id':3, 'trees':0, 'strengths':[1,1,3], 'actions':['trees','trees','trees'], 'blames':['medium','medium','high'],   'unknowns':['trees']},   # implies n_trees=1
	{'id':4, 'trees':0, 'strengths':[3,1,1], 'actions':['trees','trees','trees'], 'blames':['high','low','low'],         'unknowns':['trees']},		    # implies n_trees=2
	{'id':5, 'trees':0, 'strengths':[1,3,1], 'actions':['trees','trees','trees'], 'blames':['high','low','high'],        'unknowns':['trees']},        # implies n_trees=3
	{'id':6, 'trees':2, 'strengths':[0,2,3], 'actions':['trees','trees','fish'],  'blames':['high','low','low'],         'unknowns':['strength_a']},          # implies 1 or 3 strength_a
	{'id':7, 'trees':2, 'strengths':[3,0,2], 'actions':['fish','trees','trees'],  'blames':['low','medium','medium'],    'unknowns':['strength_b']},     # implies 2 strength_a
	{'id':8, 'trees':2, 'strengths':[3,3,0], 'actions':['trees','fish','trees'],  'blames':['medium','medium','high'],   'unknowns':['strength_c']},    # implies 1 strength_a
	{'id':9, 'trees':2, 'strengths':[3,0,3], 'actions':['trees','trees','fish'],  'blames':['high','low','low'],         'unknowns':['strength_b']},          # implies 2 strength_a
	{'id':10,'trees':2, 'strengths':[3,3,0], 'actions':['fish','trees','trees'],  'blames':['low','high','high'],        'unknowns':['strength_c']},         # implies 3 strength_a
	{'id':11,'trees':1, 'strengths':[1,3,2], 'actions':[None,'trees','fish'],     'blames':['low','high','low'],         'unknowns':['action_a']},             # implies A trees
	{'id':12,'trees':1, 'strengths':[1,2,3], 'actions':[None,'fish','trees'],     'blames':['high','low','high'],        'unknowns':['action_a']},            # implies A fished
	{'id':13,'trees':3, 'strengths':[3,2,2], 'actions':['fish',None,'fish'],      'blames':['high','medium','low'],      'unknowns':['action_b']},           # implies A trees
	{'id':14,'trees':3, 'strengths':[2,3,2], 'actions':['fish','fish',None],      'blames':['low','high','low'],         'unknowns':['action_c']},              # implies A fished
	{'id':15,'trees':3, 'strengths':[3,1,2], 'actions':['fish',None,'trees'],     'blames':['medium','medium','medium'], 'unknowns':['action_b']},     # implies A trees
	{'id':16,'trees':3, 'strengths':[2,3,1], 'actions':['trees','fish',None],     'blames':['medium','medium','high'],   'unknowns':['action_c']},       # implies A fished
	{'id':17,'trees':0, 'strengths':[1,2,1], 'actions':[None,'fish',None],        'blames':['high','low','high'],        'unknowns':['trees', 'action_a', 'action_c']},               # 1 tree, A(A,B)=fish very tricky
	{'id':18,'trees':3, 'strengths':[0,0,3], 'actions':['trees','trees','trees'], 'blames':['medium','medium','low'],    'unknowns':['strength_a', 'strength_b']},    # A and B prob had sub-3 strengths to be blamed for trees
	{'id':19,'trees':3, 'strengths':[1,0,1], 'actions':['fish',None,'fish'],      'blames':['low','high','low'],         'unknowns':['strength_b', 'action_b']},              # A almost definitely has 3 strength and went fishing
	{'id':20,'trees':3, 'strengths':[2,1,0], 'actions':['fish','fish','fish'],    'blames':['low','low','high'],         'unknowns':['strength_c']},            # A prob has 3 strength and should have cleared trees
	{'id':21,'trees':3, 'strengths':[1,1,2], 'actions':[None,None,'fish'],        'blames':['low','low','high'],         'unknowns':['action_a', 'action_b']},                # Pivotality: one of A or B fished the other trees, rationality assigns medium to everyone
	{'id':22,'trees':0, 'strengths':[1,2,2], 'actions':['trees','trees','trees'], 'blames':['low','medium','medium'],    'unknowns':['trees']},    # implies n_trees=1
	{'id':23,'trees':0, 'strengths':[2,1,2], 'actions':['trees','trees','trees'], 'blames':['medium','medium','medium'], 'unknowns':['trees']}, # implies n_trees=2
	{'id':24,'trees':0, 'strengths':[2,2,1], 'actions':['trees','trees','trees'], 'blames':['high','high','low'],        'unknowns':['trees']},        # pivotality implies n_trees=3, rationality less committal
	{'id':25,'trees':1, 'strengths':[2,3,1], 'actions':['trees',None,'trees'],    'blames':['high','high','low'],        'unknowns':['action_b']},           # easy, A went to trees
	{'id':26,'trees':1, 'strengths':[2,1,1], 'actions':[None,'fish','fish'],      'blames':['high','medium','medium'],   'unknowns':['action_a']},        # rationality says A trees, pivotality says all medium
	{'id':27,'trees':1, 'strengths':[1,2,1], 'actions':['fish',None,'fish'],      'blames':['medium','low','medium'],    'unknowns':['action_b']},         # rationality says A fish, pivotality wants low high high
	{'id':28,'trees':3, 'strengths':[1,2,3], 'actions':['fish',None,None],        'blames':['high','medium','medium'],   'unknowns':['action_b', 'action_c']},          # pivotality says C gets lots of blame if A trees B fish <- tricky but cool
	{'id':29,'trees':3, 'strengths':[3,1,2], 'actions':[None,'fish',None],        'blames':['high','medium','medium'],   'unknowns':['action_a', 'action_c']},          # pivotality says A gets lots of blame if A and B fish <- tricky but cool
	{'id':30,'trees':2, 'strengths':[1,1,2], 'actions':[None,None,'fish'],        'blames':['high','medium','medium'],   'unknowns':['action_a', 'action_b']},          # pivotality says implies A fished and B trees
	{'id':31,'trees':2, 'strengths':[2,1,1], 'actions':['fish',None,None],        'blames':['high','medium','medium'],   'unknowns':['action_b', 'action_c']},          # pivotality says implies A and B fished, stronger than rationality
	{'id':32,'trees':3, 'strengths':[1,3,0], 'actions':['trees','fish','fish'],   'blames':['medium','medium','low'],    'unknowns':['strength_c']},      # pivotality says S(A)=1, rationality prefers S(A)=2 but ambivalent
	{'id':33,'trees':3, 'strengths':[1,0,3], 'actions':['trees','fish','fish'],   'blames':['medium','high','medium'],   'unknowns':['strength_b']},     # pivotality says S(A)=2, rationality prefers S(A)=3
	{'id':34,'trees':3, 'strengths':[3,1,0], 'actions':['fish','trees','fish'],   'blames':['medium','medium','medium'], 'unknowns':['strength_c']},   # pivotality says S(A)=3, rationality ambivalent
	{'id':35,'trees':3, 'strengths':[3,3,1], 'actions':[None,'trees','fish'],     'blames':['high','high','low'],        'unknowns':['action_a']}             # pivotality says A trees, rationality says A fished
]

# converting to a format for csv/running models
trials = pd.DataFrame.from_dict(cases)
# replacing 0 trees or strengths with None values for functions
trials.trees[trials.trees==0] = None
for e in range(trials.shape[0]):
	trials.strengths[e] = np.where(np.array(trials.strengths[e])==0, None, trials.strengths[e])
# functions expect None values, not NaN
trials = trials.where(pd.notnull(trials),None)


def blame_process(blame):
	if blame=='low':
		return 0.2
	elif blame=='medium':
		return 0.5
	elif blame=='high':
		return 0.8
	else:
		return blame
def action_process(action):
	if action=='fish':
		return 0
	elif action=='trees':
		return 1

"""
Takes a posterior distribution over situations and judgment(s), and computes the likelihood of each choice
parameters:
	post		  pd df 		posterior distribution over situations
	ppt_data	  pd df 		participant judgments on that trial
return:
	judge_lk	  dict			What situational aspect they were asked about, what choice they made, and the likelihood of that choice
"""
def judgment_selector(post, ppt_data):
	judge_lk = {'question':[], 'choice':[], 'lk':[]}
	for i in range(ppt_data.shape[0]):
		# finding the judgment a participant made
		judge_type = ppt_data.iloc[i,]['question']
		judge = ppt_data.iloc[i,]['choice']
		# storing these values and the likelihood of that judgment
		judge_lk['question'].append(judge_type)
		judge_lk['choice'].append(judge)
		judge_lk['lk'].append(np.sum(post.loc[post[judge_type]==judge]['post']))
	return judge_lk

"""
For a particular trial, computes the posterior over situations
parameters:
	decision_beta float 		softmax parameter over decisions
	case		  pd df 		knowns and unknowns about the situation
	ppt_data	  pd df 		participant judgments on that trial
	model 		  pd df 		generative model predictions of blame values (e.g. rationality, pivotality, mixture)
return (depending on flag):
	post	  	  pd df			posterior over situations
"""
def lk(decision_beta, case, ppt_data, model):
	trees = case['trees']
	strength_a = case['strengths'][0]
	strength_b = case['strengths'][1]
	strength_c = case['strengths'][2]
	action_a = action_process(case['actions'][0])
	action_b = action_process(case['actions'][1])
	action_c = action_process(case['actions'][2])
	reward = None
	blame_a = blame_process(case['blames'][0])
	blame_b = blame_process(case['blames'][1])
	blame_c = blame_process(case['blames'][2])

	# calculating posterior distribution over states
	post = situation_inf(model, 
		decision_beta=decision_beta, 
		scale=.1, trees=trees, 
		strength_a=strength_a, 
		strength_b=strength_b, 
		strength_c=strength_c, 
		action_a=action_a, 
		action_b=action_b, 
		action_c=action_c, 
		reward=reward, 
		blame_a=blame_a, 
		blame_b=blame_b, 
		blame_c=blame_c) 
	return post

"""
Helper function for softmaxing participant judgments by question
parameters:
	post		  df 			posterior distribution over situations
	beta 		  float 		softmax temperature parameter
	ppt_type 	  string 		flag specifically designed to catch participants who act to maximize reward (instead of based on blame)
return:
	post		  dict			softmaxed posterior
"""
def softmax_judge(post, beta, ppt_type='blame_sensitive'):
	variables = ['trees','strength_a','strength_b','strength_c','action_a','action_b','action_c']
	# searching over each possible unknown factor
	for variable in variables:
		if post.nunique()[variable] > 1:
			# marginalizing over other variables
			tmp = post[[variable,'post']].groupby(variable).sum()
			# accounting for participants who acted to maximize reward after seeing what everyone else did
			if ppt_type=='reward_maximizer' and 'action' in variable:
				tmp = post[[variable,'reward']].groupby(variable).sum()
				tmp.rename(columns = {'reward': 'post_' + variable}, inplace=True)
			# softmaxing
			tmp2 = np.exp(tmp*beta)/np.sum(np.exp(tmp*beta))
			# cleaning up and saving
			tmp2.rename(columns = {'post':'post_' + variable}, inplace=True)
			post = post.merge(tmp2, on=variable)
	# multiplying the softmaxed probabilities by each other to get one posterior distribution
	post_names = post.filter(regex='^post_',axis=1)
	post['softmax'] = post.filter(post_names, axis=1).apply(np.prod, axis=1)
	post.drop(['post'], axis=1, inplace=True)
	post.drop(list(post.filter(regex = '^post_')), axis=1, inplace=True)
	post.rename(columns = {'softmax':'post'}, inplace=True)
	return post

"""
Qualitative model predictions for specific parameters
Posterior distributions for each trial, for each judgment within that trial. Useful for plotting model predictions against people
parameters:
	model_posts   list of dfs 	list of posteriors for each model (rationality, pivotality, mixture)
	decision_beta array 		softmax temperature parameter
	w   	 	  float 		weight between rationality and pivotality models (1 full rationality)
return:
	exp_post	  dict			posterior over choices
sample call:
	model_qual_preds(model_posts=[rationality_posts, pivotality_posts], decision_beta=[1, .5, 1, .25], w=.8)
"""
def model_qual_preds(model_pars, ppt_data):
	model_store = []
	post_store = []
	for model_par in model_pars:
		# getting blame values for this model
		model_blame = situations_df(n_trees=3, n_strengths=3, n_actions=2, n_people=3, w=model_par['w'], rationality_beta=model_par['rationality_beta'], k=model_par['k'])
		model_store.append(model_blame)
		for trial_idx in range(trials.shape[0]):
			# pick out the relevant case/trial to consider
			case = trials.iloc[trial_idx,:]
			ppt_data = None
			# compute the likelihood of each option
			post = lk(decision_beta=model_par['decision_beta'], case=case, ppt_data=ppt_data, model=model_blame)
			post['model'] = model_par['model_name']
			post['trial_id'] = trial_idx
			post_store.append(post)

			# reward maximizers
			if 0:
				model_rewardmaxer = softmax_judge(post_store[0], beta=decision_beta[3], ppt_type='reward_maximizer')
				model_rewardmaxer['model'] = 'reward_maximizer'
				model_rewardmaxer['trial_id'] = trial_idx
				post_store.append(model_rewardmaxer)

				# random model
				model_random = softmax_judge(post_store[0], beta=0)
				model_random['model'] = 'random'
				model_random['trial_id'] = trial_idx
				post_store.append(model_random)
	exp_posts = pd.concat(post_store)
	exp_posts.to_csv('../../../data/cogsci2021/inference_trials2.csv', index=False)
	return exp_posts
# qualitative model predictions for best-fitting models
if 0:
	fitted_pars = [{'model_name': 'rationality', 'w': 1, 'rationality_beta': 1, 'k': 2, 'decision_beta': 1.5},
					{'model_name': 'pivotality', 'w': 0, 'rationality_beta': 1, 'k': 1, 'decision_beta': 1.5},
					{'model_name': 'mixture', 'w': .9, 'rationality_beta': 1, 'k': 2, 'decision_beta': 1.5},
					{'model_name': 'random', 'w': 1, 'rationality_beta': 0, 'k': 1, 'decision_beta': 0}]
	tmp = model_qual_preds(model_pars=fitted_pars, ppt_data=ppt_data)
	
"""
returns the likelihood of all participant judgments for a given model 
parameters:
	blame_model 	pd df 		dataframe of blame assignments for each situation
	pars 			dict 		model parameters
	ppt_df 			pd df 		participant choices per trial
return:
	store 			pd df 		participant choices, model parameters, and the likelihood of that choice
"""
def model_lk(blame_model, pars, ppt_df):
	store = {'participant':[], 'trial':[], 'question':[], 'choice':[], 'w':[], 'k':[], 'rationality_beta':[], 'decision_beta':[], 'lk':[]}
	# pre-computing this model's posterior for each case
	posts_store = []
	ppt_run = ppt_df[ppt_df.participant==1]
	for trial in ppt_run.trial.unique():
		ppt_data = ppt_run[ppt_run.trial==trial]
		case = trials[(trials.id+1)==trial].iloc[0]
		model_inf = lk(decision_beta=pars['decision_beta'], case=case, ppt_data=ppt_data, model=blame_model)
		model_inf = softmax_judge(model_inf, beta=pars['decision_beta'])
		model_inf = model_inf.rename(columns = {'strength_a':'a_strength', 'strength_b':'b_strength', 'strength_c':'c_strength', 'action_a':'a_choice', 'action_b':'b_choice', 'action_c':'c_choice'})
		model_inf['trial'] = trial
		posts_store.append(model_inf)
	posts = pd.concat(posts_store)

	# getting the likelihood of each participant's judgments for this model
	for ppt_id in ppt_df.participant.unique():
		ppt_run=ppt_df[ppt_df.participant==ppt_id]
		for trial in ppt_run.trial.unique():
			ppt_data = ppt_run[ppt_run.trial==trial]
			case = trials[(trials.id+1)==trial].iloc[0]
			# generating unsoftmaxed model predictions
			judge_lk = judgment_selector(post=posts[posts.trial==trial], ppt_data=ppt_data)
			for e in range(ppt_data['question'].shape[0]):
				store['participant'].append(ppt_id)
				store['trial'].append(trial)
				store['question'].append(judge_lk['question'][e])
				store['choice'].append(judge_lk['choice'][e])
				store['w'].append(pars['w'])
				store['k'].append(pars['k'])#model['k'].unique().item())
				store['rationality_beta'].append(pars['rationality_beta'])#model['rationality_beta'].unique().item())
				store['decision_beta'].append(pars['decision_beta'])
				store['lk'].append(judge_lk['lk'][e])
	return pd.DataFrame(store)

"""
mixing rationality and pivotality blame values
parameters:
	models_blame 	pd df 		dataframe that contains blame values for the rationality and pivotality models
	w 				float 		weight on rationality model
return:
	models_mix 		pd df 		dataframe mixing blame values
"""
def mix_models(models_blame, w):
	models_mix = models_blame.copy()

	# adding the mixing weight and computing mixed blames
	models_mix['w'] = w
	models_mix['blame_a'] = w*models_mix['blame_a_rat'] + (1-w)*models_mix['blame_a_piv']
	models_mix['blame_b'] = w*models_mix['blame_b_rat'] + (1-w)*models_mix['blame_b_piv']
	models_mix['blame_c'] = w*models_mix['blame_c_rat'] + (1-w)*models_mix['blame_c_piv']
	
	return models_mix


"""
mixing generative components and performing inference
parameters:
	w 				float 		mixture of rationality and pivotality
	mixture_search 	dict 		parameter combinations to do inference over
	models_blame 	pd df 		base pivotality and rationality models
	ppt_df 			pd df 		participant choices
return:
	store 			pd df 		
"""
def models_inf(mixture_search, models_blame, ppt_df):
	store = []
	for w in mixture_search['w']:
		pars = {'w':w}
		# mixing rationality and pivotality models by w
		blame_model = mix_models(models_blame, w)
		for rationality_beta in mixture_search['rationality_beta']:
			pars['rationality_beta'] = rationality_beta
			for k in mixture_search['k']:
				pars['k'] = k
				# filtering out mixed posteriors to only take this model's blame values
				tmp_blame_model = blame_model[(blame_model['k']==k) & (blame_model['rationality_beta']==rationality_beta)]
				for decision_beta in mixture_search['decision_beta']:
					pars['decision_beta'] = decision_beta
					print(pars)
					# likelihood of participant choices for this parameter
					tmp = model_lk(tmp_blame_model, pars, ppt_df)
					date = datetime.now()
					tmp.to_csv('./intermediate_analyses/gen_mix_posts3' + str(date) + '.csv')
					store.append(tmp)
	return pd.concat(store)

"""
Running the whole experiment. 
Warning! This takes a long time
parameters:
	ppt_df 			pd df 		participant judgments
	model_type 		string 		flag for different pivotality models
	posts_name 		string 		what to save the file as
return:
	Saves the csv
"""
def exp1runner(ppt_df, model_type, posts_name):
	rat_blame = models_grid[models_grid['w']==1]
	if model_type=='allen':
		piv_blame = models_grid[models_grid['w']==0].filter(['situation','blame_a','blame_b','blame_c'])
	elif model_type == 'graded':
		piv_blame  = situations_df(n_trees=3, n_strengths=3, n_actions=2, n_people=3, w=0, rationality_beta=1, k=1, graded_piv=True)
		piv_blame.blame_a = piv_blame.blame_a/6
		piv_blame.blame_b = piv_blame.blame_b/6
		piv_blame.blame_c = piv_blame.blame_c/6
		piv_blame['situation'] = piv_blame.index
	models_blame = rat_blame.merge(piv_blame, on='situation', how='left')
	models_blame.rename(columns={'blame_a_x':'blame_a_rat', 'blame_b_x':'blame_b_rat', 'blame_c_x':'blame_c_rat', 'blame_a_y':'blame_a_piv', 'blame_b_y':'blame_b_piv', 'blame_c_y':'blame_c_piv'}, inplace=True)
	start = time.time()
	posts = models_inf(mixture_search, models_blame, ppt_df)
	end = time.time(); print(end-start)
	posts.to_csv('./intermediate_analyses/' + posts_name + '.csv')

# running the whole experiment
if 0:
	models_blame = rat_blame.merge(piv_blame, on='situation', how='left')
	mixture_search = {'w': [0, .1, .3, .5, .7, .9, 1], 'decision_beta': [.25, .5, 1, 1.5, 3, 9], 'rationality_beta': models_blame.rationality_beta.unique(), 'k': models_blame.k.unique()}
	start = time.time()
	a = models_inf(mixture_search, models_blame, ppt_df)
	end = time.time(); print(end-start)
	a.to_csv('./intermediate_analyses/gen_mix_posts.csv')
































