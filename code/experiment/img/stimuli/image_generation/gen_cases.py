"""
Generating cases from base images

import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# or a hacky version of this
os.chdir(os.path.expanduser('~') + '/Documents/inference_from_blame/code/experiments/experiment_inference/img/stimuli/image_generation')  
"""

# general setup ------------------------------
def situation_generator(n_trees, strengths, actions, blames):
	base = Image.open('base_images/base.png')

	# trees
	tree = Image.open('misc/tree.png')
	tree.thumbnail((55,55))

	# fish
	"""
	if n_fish == 0:
		fish = Image.open('misc/no_fish.png'); fish.thumbnail((55,55))
	else:
		fish = Image.open('misc/fish.png'); fish.thumbnail((45,45))
	"""

	# strengths
	A_strength = Image.open('strengths/A.png'); A_strength.thumbnail((13,13))
	B_strength = Image.open('strengths/B.png');	B_strength.thumbnail((13,13))
	C_strength = Image.open('strengths/C.png');	C_strength.thumbnail((13,13))
	strengths_vec = [A_strength, B_strength, C_strength]
	strengths_loc = [[160,212], [550,212], [448,352]]

	# actions: a hassle because arrow sizes/locations must be handled separately
	axe = Image.open('actions/axe.png'); axe.thumbnail((40,40))
	rod = Image.open('actions/rod.png'); rod.thumbnail((60,60))
	# if there's an action A, draw an arrow and place it at the right place (otherwise do nothing)
	if actions[0] is not None:
		A_arrow = Image.open('actions/arrows/A_' + actions[0] + '.png')
		if actions[0]=='trees':
			A_action = axe
			A_arrow.thumbnail((100,100))
			A_loc = [195,162]
		elif actions[0]=='fish':
			A_action = rod
			A_arrow.thumbnail((70,70))
			A_loc = [85,122]
		# action box location does not change
		A_action_loc = [160,242]
		base.paste(A_action, (A_action_loc), A_action)
		base.paste(A_arrow, (A_loc), A_arrow)

	if actions[1] is not None:
		B_arrow = Image.open('actions/arrows/B_' + actions[1] + '.png')
		if actions[1]=='trees':
			B_action = axe
			B_arrow.thumbnail((100,100))
			B_loc = [410,162]
		elif actions[1]=='fish':
			B_action = rod
			B_arrow.thumbnail((70,70))
			B_loc = [555,122]
		B_action_loc = [550,242]
		base.paste(B_action, (B_action_loc), B_action)
		base.paste(B_arrow, (B_loc), B_arrow)

	if actions[2] is not None:
		C_arrow = Image.open('actions/arrows/C_' + actions[2] + '.png')
		if actions[2]=='trees':
			C_action = axe
			C_arrow.thumbnail((60,60))
			C_loc = [351,236]
		elif actions[2]=='fish':
			C_action = rod
			C_arrow.thumbnail((70,70))
			C_loc = [266,356]
		C_action_loc = [448,381]
		base.paste(C_action, (C_action_loc), C_action)
		base.paste(C_arrow, (C_loc), C_arrow)

	# blame
	A_blame = Image.open('blame/A_' + blames[0] + '.png'); A_blame.thumbnail((135,135))
	B_blame = Image.open('blame/B_' + blames[1] + '.png'); B_blame.thumbnail((135,135))
	C_blame = Image.open('blame/C_' + blames[2] + '.png'); C_blame.thumbnail((135,135))
	blames_vec = [A_blame, B_blame, C_blame]
	blames_loc = [[90,266], [480,266], [378,405]]

	# adding trees to base
	for tree_idx in range(n_trees):
		base.paste(tree, (330, 150 - (40*tree_idx)), tree)

	# adding strengths, actions, and blames per px
	for agent in range(len(strengths)):
		for idx in range(strengths[agent]):
			# adding strengths
			base.paste(strengths_vec[agent], (strengths_loc[agent][0] + (16*idx),strengths_loc[agent][1]), strengths_vec[agent])
		# adding blames
		base.paste(blames_vec[agent], (blames_loc[agent]), blames_vec[agent])


	# DEPRECATED: adding fish params: grabs base, location, number, color, font
	#font = ImageFont.truetype("fonts/helveticaneue/HelveticaNeue BlackCond.ttf", 32)
	#ImageDraw.Draw(base).text((688,393), str(n_fish), (0,0,0), font=font)

	#return(base)
	base = base.resize((900,552), Image.ANTIALIAS)
	name = 't' + str(n_trees) + '_A' + str(strengths[0]) + str(actions[0]) + str(blames[0]) + '_B' + str(strengths[1]) + str(actions[1]) + str(blames[1]) + '_C' + str(strengths[2]) + str(actions[2]) + str(blames[2])
	base.save('../images/' + name + '.png', 'png')
	return(base)

# parameters ------------------------------
n_trees= 3
strengths = np.array([0,1,1])
actions = np.array(['fish','fish','fish'])
blames = ['high','low','low']

gum = situation_generator(n_trees=n_trees, strengths=strengths, actions=actions, blames=blames)
gum.show()

#gum.save('../images/test', 'png')











