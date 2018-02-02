
# A Spiking Neural Model for Learning Priors

In this paper, we present a spiking neural model of learning priors for a lifespan inference task. Through this model, we extend our previous work on the biological plausibility of performing Bayesian computations in the brain. Specifically, we address the question of how humans might be generating the priors needed for this inference. We propose neural mechanisms that may be involved in continuous learning and updating of priors and may be applicable in many aspects of higher-level cognition. We show that the model is generalizable and is able to converge to the near optimal priors with very few training samples. 



### Source Code
- See `model/prior_learning_model.ipynb.ipynb` for the neural model implementation and the theoretical groundwork.

### Results
- The notebook `model/results.ipynb` uses the pickle files `27fspace.p` or `108fspace.p` to load the prior spaces. The pickle files containing simulation results are not uploaded (they are huge), but can be provided on request.

### Paper
- See `latexpaper` for uncompiled LaTeX paper.

### Running the Model
- Run the  `model/script.py` from the terminal in order to generate simulation results for specified number of trials and training observations (arguments to the script). 
- Output would a pickle file corresponding to each trial that can then be probed at different points to get data corresponding to different number of training observations.


### Some Dependencies
- Python
- Scipy
- Nengo
- Numpy
- Seaborn
- Jupyter
- Matplotlib
