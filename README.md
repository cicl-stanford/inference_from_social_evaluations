# Project name

## General points

This repository contains the data, materials, figures, and analysis script for the paper "Who went fishing? Inferences from social evaluations" by Zachary J. Davis, Kelsey R. Allen, and Tobias Gerstenberg.

Feel free to reach out in case you have any questions about the repository (email: zach.davis@stanford.edu).

The paper can be accessed [here](https://psyarxiv.com/x4zav)

## Abstract 

Humans have a remarkable ability to go beyond the observable. From seeing the current state of our shared kitchen, we can infer what happened and who did it. Prior work has shown how the physical state of the world licenses inferences about the causal history of events, and the agents that participated in these events. Here, we investigate a previously unstudied source of evidence about what happened: social evaluations. In our experiment, we present situations in which a group failed to optimally coordinate their actions. Participants learn how much each agent was blamed for the outcome, and their task is to make inferences about the situation, the agents' actions, as well as the agents' capabilities. We develop a computational model that accurately captures participants' inferences. The model assumes that people blame others by considering what they should have done, and what causal role their action played. By inverting this generative model of blame, people can figure out what happened.

## Repository structure 

```
.
├── code
│   ├── R
│   ├── experiment
│   └── python
├── data
│   └── model_results
├── docs
└── figures
    ├── plots
    └── stimuli
```

- `code/R/`: RMarkdown document for descriptives and visualizations.
	+ You can view a rendered html file of the analysis [here](https://cicl-stanford.github.io/inference_from_social_evaluations/). 
- `code/experiment/`: experiment code using [psiturk](https://psiturk.org/)
- `code/python/`: modeling code and analysis
	+ `model/`:
		- `action_blame.py`: generative model of blame
		- `blame_inference.py`: inference over situations
	+ `analysis/`:
		- `experiment_inference_analysis.py`: model comparisons on participant judgments
- `data/`:
	+ `ppt_df.csv`: raw data
	+ `df.long.csv`: processed data describing each participant's choices in each trial