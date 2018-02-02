
# A Spiking Neural Model for Learning Priors

In this paper, we present a spiking neural model of learning priors for a life span inference task. Through this model, we extend our previous work on the biological plausibility of performing Bayesian computations in the brain. Specifically, we address the question of how humans might be generating the priors needed for this inference. We propose neural mechanisms for continuous learning and updating of priors - cognitive processes that are critical for many aspects of higher-level cognition. The model is generalizable to many different psychological tasks and we show that it is able to converge to the near optimal priors with very few training examples. 



### Source Code
- See `model/prior_learning_model.ipynb.ipynb` for the neural model implementation and the theoretical groundwork.

### Results
- The notebook `model/results.ipynb` uses the pickle files `27fspace.p` or `108fspace.p` to load the prior spaces. The pickle files containing simulation results are not uploaded (they are huge), but can be provided on request.

### Paper
- See `latexpaper` for uncompiled LaTeX paper.

### Running the Model
- Run `model/script.py` from the terminal in order to generate simulation results for specified number of trials and training observations (arguments to the script). 
- Output would a pickle file corresponding to each trial that can then be probed at different points to get data corresponding to different number of training observations.


## Dependencies

Version numbers state the versions used to generate the results. Newer and older versions might work as well, but have not been tested.

### General dependencies
- [Python 2.7.9](https://www.python.org/)
- [Nengo 2.6.1](https://github.com/nengo/nengo)
- [Numpy 1.12.1](http://www.numpy.org/)
- [Scipy 0.18.1](https://www.scipy.org/)


### Jupyter notebook
- [Jupyter](http://jupyter.org/)
  - jupyter 1.0.0
  - jupyter-client 4.2.2
  - jupyter-core 4.1.0
  - jupyter-console 4.1.1
- [Matplotlib 1.4.3](http://matplotlib.org/)
- [Seaborn 0.7.1](http://seaborn.pydata.org/)


### Visualization
- [Nengo GUI 0.2.1](https://github.com/nengo/nengo_gui)

