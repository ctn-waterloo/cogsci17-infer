import nengo
import numpy as np
import scipy.special as sp
import scipy.stats as st
import cPickle as pickle


import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace
import sys

import os
import sys
if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file to store the output data"
    exit(0)


max_age = dim = 120

# our domain is thetas (i.e., age from 1 to 120)
thetas = np.linspace(start=1, stop=max_age, num=max_age)


# prior parameters
skew = -6
loc = 99
scale = 27

def likelihood(x):
    x = int(x)
    like = np.asarray([1/p for p in thetas])
    like[0:x-1] = [0]*np.asarray(x-1)
    return like

def skew_gauss(skew, loc, scale):
    return [(st.skewnorm.pdf(p, a=skew, loc=loc, scale=scale)) for p in thetas] 
    
def posterior(x, skew, loc, scale):
    post = likelihood(x=x)*skew_gauss(skew=skew, loc=loc, scale=scale)
    return post

def normalized_posterior(x, skew, loc, scale):
    post = posterior(x, skew, loc, scale)
    post = post/sum(post)
    return post



ages = np.linspace(start=1, stop=100, num=100, dtype=np.int32)
data = {}
n_basis = 20
for x in ages:
    if x<5:
        pad = 5-x+1
    else:
        pad = 0
       
    # define sub-spaces
    space = nengo.FunctionSpace(
            nengo.dists.Function(skew_gauss,
                             skew=nengo.dists.Uniform(skew-1, skew+2), 
                          loc=nengo.dists.Uniform(loc-1,loc+2), 
                          scale=nengo.dists.Uniform(scale-1, scale+2)),
                            n_basis=n_basis)

    from copy import deepcopy
    space_raw = deepcopy(space.space)


    lik_space = nengo.FunctionSpace(
                    nengo.dists.Function(likelihood,
                                x=nengo.dists.Uniform(x-5+pad,x+5+pad)),
                    n_basis=n_basis)
    
    lik_space_raw = deepcopy(lik_space.space)

    post_space = nengo.FunctionSpace(
                    nengo.dists.Function(posterior,
                                 x=nengo.dists.Uniform(x-5,x+5),
                                skew=nengo.dists.Uniform(skew-1, skew+2), 
                              loc=nengo.dists.Uniform(loc-50,loc+2), 
                              scale=nengo.dists.Uniform(scale-1, scale+2)),
                    n_basis=n_basis)
    
    post_space_raw = deepcopy(post_space.space)

    norm_post_space = nengo.FunctionSpace(
                nengo.dists.Function(normalized_posterior,
                             x=nengo.dists.Uniform(x-5+pad,x+5+pad),
                            skew=nengo.dists.Uniform(skew-1, skew+2), 
                          loc=nengo.dists.Uniform(loc-50,loc+2), 
                          scale=nengo.dists.Uniform(scale-1, scale+2)),
                n_basis=n_basis)

    norm_post_space_raw = deepcopy(norm_post_space.space)

    # Nengo model
    k = np.zeros((120, n_basis))    # post basis for reconstruction     
    j = 0
    for element in space.basis.T:
        a = np.multiply(element, lik_space.basis.T[j])
        k[:, j] = a 
        j = j + 1        

    post_space._basis = k
    model = nengo.Network()
    #model.config[nengo.Ensemble].neuron_type=nengo.Direct()
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
                                 neuron_type = nengo.Direct())
        product = nengo.networks.Product(n_neurons=100*2, dimensions=post_space.n_basis, input_magnitude=1)
        
        nengo.Connection(ens, product.A)
        nengo.Connection(ens2, product.B)
        nengo.Connection(product.output, post)
        probe_func3 = nengo.Probe(post, synapse=0.03)
        
        # normalization
        def normalize(a):
            b = np.dot(a, k.T)
            total = np.sum(b)
            if total == 0:
                return [0]*dim
            return b/total
        
        
        # Note: this population needs to have around 250 neurons for accurate representation
        norm_post = nengo.Ensemble(label="Normalized Posterior", n_neurons=800, dimensions=dim, 
                                   encoders=norm_post_space_raw,
                                 eval_points=norm_post_space_raw)
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

pickle.dump(data, open(fname, 'wb'))
print("pickle complete")
print(fname)
