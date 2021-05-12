# Project name

## General points

This repository contains the data, materials, figures, and analysis script for the paper "Who went fishing? Inferences from social evaluations" by Zachary J. Davis, Kelsey Allen, and Tobias Gerstenberg.

Feel free to reach out in case you have any questions about the repository (email: zach.davis@stanford.edu).

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