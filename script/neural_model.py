import nengo
import numpy as np
import scipy.special as sp
import scipy.stats as st
import cPickle as pickle


import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace
import sys


max_age = dim = 120

# our domain is thetas (i.e., age from 1 to 120)
thetas = np.linspace(start=1, stop=max_age, num=max_age)


# prior parameters
skew = -4
loc = 97
scale = 28

def likelihood(x):
    x = int(x)
    like = np.asarray([1/p for p in thetas])
    like[0:x-1] = [0]*np.asarray(x-1)
    return like


def skew_gauss(skew, loc, scale):
    return [(st.skewnorm.pdf(p, a=skew, loc=loc, scale=scale)) for p in thetas] 
    
    
def posterior(x, skew, loc, scale):
    post = likelihood(x=x)*skew_gauss(skew=skew, loc=loc, scale=scale)
    post = post/sum(post)
    return post


try:
    ages = np.linspace(start=1, stop=100, num=100, dtype=np.int32)
    data = {}
    for x in ages:
        # define sub-spaces
        space = nengo.FunctionSpace(
                nengo.dists.Function(skew_gauss,
                                 skew=nengo.dists.Uniform(skew-1, skew+2), 
                              loc=nengo.dists.Uniform(loc-1,loc+2), 
                              scale=nengo.dists.Uniform(scale-1, scale+2)),
                                n_basis=50)
    
        from copy import deepcopy
        space_raw = deepcopy(space.space)
    
    
        lik_space = nengo.FunctionSpace(
                        nengo.dists.Function(likelihood,
                                    x=nengo.dists.Uniform(x-1,x+2)),
                        n_basis=50)
        
        lik_space_raw = deepcopy(lik_space.space)
    
        post_space = nengo.FunctionSpace(
                        nengo.dists.Function(posterior,
                                     x=nengo.dists.Uniform(x-1,x+2),
                                    skew=nengo.dists.Uniform(skew-1, skew+2), 
                                  loc=nengo.dists.Uniform(loc-1,loc+2), 
                                  scale=nengo.dists.Uniform(scale-1, scale+2)),
                        n_basis=50)
        
        post_space_raw = deepcopy(post_space.space)
    
    
    
        # Nengo model
        model = nengo.Network(seed=12)
        with model:
            stim = nengo.Node(label="prior input", output=space.project(skew_gauss(skew=skew, loc=loc, scale=scale)))
            ens = nengo.Ensemble(label="Prior", n_neurons=200, dimensions=space.n_basis,
                                 encoders=space.project(space_raw),
                                 eval_points=space.project(space_raw),
                                )
            
            stim2 = nengo.Node(label="likelihood input", output=lik_space.project(likelihood(x=x)))
            ens2 = nengo.Ensemble(label="Likelihood", n_neurons=200, dimensions=lik_space.n_basis,
                                 encoders=lik_space.project(lik_space_raw),
                                 eval_points=lik_space.project(lik_space_raw),
                                )
            
            
            nengo.Connection(stim, ens)
            probe_func = nengo.Probe(ens, synapse=0.03)
            
            nengo.Connection(stim2, ens2)
            probe_func2 = nengo.Probe(ens2, synapse=0.03)
            
            # elementwise multiplication
            post = nengo.Ensemble(label="Posterior", n_neurons=200, dimensions=post_space.n_basis,
                                     encoders=post_space.project(post_space_raw),
                                     eval_points=post_space.project(post_space_raw),
                                    )
            product = nengo.networks.Product(n_neurons=50*2, dimensions=post_space.n_basis, input_magnitude=1)
            
            nengo.Connection(ens, product.A)
            nengo.Connection(ens2, product.B)
            nengo.Connection(product.output, post)
            probe_func3 = nengo.Probe(post, synapse=0.03)
            
            # normalization
            def normalize(a):
                b = post_space.reconstruct(a)
                total = np.sum(b)
                if total == 0:
                    return [0]*dim
                return [x / total for x in b]
            
            
            # Note: this population needs to have around 250 neurons for accurate representation
            norm_post = nengo.Ensemble(label="Normalized Posterior", n_neurons=250, dimensions=dim, 
                                       encoders=post_space_raw,
                                     eval_points=post_space_raw)
            
            nengo.Connection(post, norm_post, function=normalize)
            probe_func4 = nengo.Probe(norm_post, synapse=0.03)
            
            # prediction
            def median(b):
                med = 0
                for n in np.arange(len(b)):
                    cum = sum(b[:n+1])
                    if cum == 0.5 or cum > 0.5:
                        med = n + 1
                        break
                return int(med)
        
            
            prediction = nengo.Node(label="Prediction", output=None, size_in=1)
               
            nengo.Connection(norm_post, prediction, function=median, synapse=0.03)
            probe_func5 = nengo.Probe(prediction, synapse=0.03)
    		    
    		    
        sim = nengo.Simulator(model)
        sim.run(0.5)

        node_prediction = sim.data[probe_func5][-1][0]
        data[x] = [0, node_prediction]

except:
	print "SS - Exception occured"	

finally:
	fname = 'neural_predictions.p'
	pickle.dump(data, open(fname, 'wb'))
	print("pickle complete")
	print(fname)
