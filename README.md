# A Spiking Neural Model of Life Span Inference

In this paper, we present a spiking neural model of life span inference. Through this model, we explore the biological plausibility of performing Bayesian computations in the brain. Specifically, we address the issue of representing probability distributions using neural circuits and combining them in meaningful ways to perform inference.  We show that applying these methods to the life span inference task matches human performance on this task better than an ideal Bayesian model.  We also describe potential ways in which humans might be generating the priors needed for this inference. This provides an initial step towards better understanding how Bayesian computations may be implemented in a biologically plausible neural network. 



### Source Code
- See `model/neural_model.ipynb` for the model implementation.
- See `model/generalized_lifespan_inference.ipynb` for theoretical groundwork on generating priors.

### Paper
- See `latexpaper` for uncompiled LaTeX paper.

### Running the Model
- Run the  `script/generate_data.py` from the terminal. It will inturn call `script/neural_model.py` to generate specified number of samples.
- Output would be the same number of pickle files.

### Results
- The notebook `results/results-plots.ipynb` uses the pickle files in the `results` folder and `results/data` folder to plot results.
- K-S-dissimilarity.csv shows the dissimilarity calculations for Kolmogorov-Smirnov (K-S) test. 


### Some Dependencies
- Python
- Scipy
- Nengo
- Numpy
- Seaborn
- Jupyter
- Matplotlib





