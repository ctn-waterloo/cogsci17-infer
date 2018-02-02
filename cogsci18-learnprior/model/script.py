# setup the environment
import nengo
import numpy as np
import scipy.stats as st
from nengo.networks import EnsembleArray
import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

import sys
import cPickle as pickle

# get the arguments
fname_input = sys.argv[1]
fname_output = sys.argv[2]
training_size = int(sys.argv[3])
num_iterations = int(sys.argv[4])
# --------------------------------------------------------------------------------- #    

def IA(d, n_neurons, dt, share_thresholding_intercepts=False):
    bar_beta = 2.  # should be >= 1 + max_input * tau2 / tau1
    tau_model1 = 0.1
    tau_model2 = 0.1
    tau_actual = 0.1

    # dynamics put into continuous LTI form:
    #   dot{x1} = A1x1 + A2x2 + Bu
    # where x1 is the state variable for layer 1 and
    #       x2 is the state variable for layer 2
    # note that from the perspective of Principle 3, A2x2 is treated
    # as an "input" similar to u
    I = np.eye(d)
    inhibit = 1 - I
    B = 1. / tau_model1  # input -> layer 1
    A1 = 0  # (integrator) layer1 -> layer1
    A2 = (I - bar_beta * inhibit) / tau_model2  # layer 2 -> layer 1

    n_neurons_threshold = 50 
    n_neurons_x = n_neurons - n_neurons_threshold
    assert n_neurons_x > 0
    threshold = 0.8

    with nengo.Network(label="IA") as net:
        net.input = nengo.Node(size_in=d)
        x = nengo.networks.EnsembleArray(
            n_neurons_x, d,
            eval_points=nengo.dists.Uniform(0., 1.),
            intercepts=nengo.dists.Uniform(0., 1.),
            encoders=nengo.dists.Choice([[1.]]), label="Layer 1")
        net.x = x
        nengo.Connection(x.output, x.input, transform=tau_actual * A1 + I,
                         synapse=tau_actual)

        nengo.Connection(
            net.input, x.input,
            transform=tau_actual * B,
            synapse=tau_actual)

        with nengo.presets.ThresholdingEnsembles(0.):
            thresholding = nengo.networks.EnsembleArray(
                n_neurons_threshold, d, label="Layer 2")
            if share_thresholding_intercepts:
                for e in thresholding.ensembles:
                    e.intercepts = nengo.dists.Exponential(
                        0.15, 0., 1.).sample(n_neurons_threshold)
            net.output = thresholding.add_output('heaviside', lambda x: x > 0.)

        bias = nengo.Node(1., label="Bias")

        nengo.Connection(x.output, thresholding.input, synapse=0.005)
        nengo.Connection(
            bias, thresholding.input, transform=-threshold * np.ones((d, 1)))
        nengo.Connection(
            thresholding.heaviside, x.input,
            transform=tau_actual * A2, synapse=tau_actual)

    return net



def InputGatedMemory(n_neurons, n_neurons_diff, dimensions, feedback=1.0,
                     difference_gain=1.0, recurrent_synapse=0.1,
                     difference_synapse=None, net=None, **kwargs):
    """Stores a given vector in memory, with input controlled by a gate.
    Parameters
    ----------
    n_neurons : int
        Number of neurons per dimension in the vector.
    dimensions : int
        Dimensionality of the vector.
    feedback : float, optional (Default: 1.0)
        Strength of the recurrent connection from the memory to itself.
    difference_gain : float, optional (Default: 1.0)
        Strength of the connection from the difference ensembles to the
        memory ensembles.
    recurrent_synapse : float, optional (Default: 0.1)
    difference_synapse : Synapse (Default: None)
        If None, ...
    kwargs
        Keyword arguments passed through to ``nengo.Network``.
    Returns
    -------
    net : Network
        The newly built memory network, or the provided ``net``.
    Attributes
    ----------
    net.diff : EnsembleArray
        Represents the difference between the desired vector and
        the current vector represented by ``mem``.
    net.gate : Node
        With input of 0, the network is not gated, and ``mem`` will be updated
        to minimize ``diff``. With input greater than 0, the network will be
        increasingly gated such that ``mem`` will retain its current value,
        and ``diff`` will be inhibited.
    net.input : Node
        The desired vector.
    net.mem : EnsembleArray
        Integrative population that stores the vector.
    net.output : Node
        The vector currently represented by ``mem``.
    net.reset : Node
        With positive input, the ``mem`` population will be inhibited,
        effectively wiping out the vector currently being remembered.
    """
    if net is None:
        kwargs.setdefault('label', "Input gated memory")
        net = nengo.Network(**kwargs)
    else:
        warnings.warn("The 'net' argument is deprecated.", DeprecationWarning)

    if difference_synapse is None:
        difference_synapse = recurrent_synapse

    n_total_neurons = n_neurons * dimensions
    n_total_neurons_diff = n_neurons_diff * dimensions

    with net:
        # integrator to store value
        
        mem_net = nengo.Network()
        mem_net.config[nengo.Ensemble].encoders = nengo.dists.Choice([[-1.]]) 
        mem_net.config[nengo.Ensemble].radius = 1
        mem_net.config[nengo.Ensemble].eval_points=nengo.dists.Uniform(-1, 0.0) 
        mem_net.config[nengo.Ensemble].intercepts = nengo.dists.Uniform(-0.6, 1.)
        with mem_net:
            net.mem = EnsembleArray(n_neurons, dimensions, label="mem")
        nengo.Connection(net.mem.output, net.mem.input,
                         transform=feedback,
                         synapse=recurrent_synapse)

        
        diff_net = nengo.Network()
        diff_net.config[nengo.Ensemble].radius = 0.5
        diff_net.config[nengo.Ensemble].eval_points=nengo.dists.Uniform(-0.5, 0.5)
        with diff_net:
            # calculate difference between stored value and input
            net.diff = EnsembleArray(n_neurons_diff, dimensions, label="diff")
        nengo.Connection(net.mem.output, net.diff.input, transform=-1)

        # feed difference into integrator
        nengo.Connection(net.diff.output, net.mem.input,
                         transform=difference_gain,
                         synapse=difference_synapse)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        net.gate = nengo.Node(size_in=1)
        net.diff.add_neuron_input()
        nengo.Connection(net.gate, net.diff.neuron_input,
                         transform=np.ones((n_total_neurons_diff, 1)) * -10,
                         synapse=None)

        # reset input (if reset=1, remove all values, and set to 0)
        net.reset = nengo.Node(size_in=1)
        nengo.Connection(net.reset, net.mem.add_neuron_input(),
                         transform=np.ones((n_total_neurons, 1)) * -3,
                         synapse=None)

    net.input = net.diff.input
    net.output = net.mem.output

    return net
    
# --------------------------------------------------------------------------------- #    
max_age = 120
thetas = np.linspace(start=1, stop=max_age, num=max_age)

# compute likelihood 
def likelihood(x):
    x = int(x)
    like = np.asarray([1/p for p in thetas])
    like[0:x-1] = [0]*np.asarray(x-1)
    return like

# computer prior
def skew_gauss(skew, loc, scale):
    prior = [(st.skewnorm.pdf(p, a=skew, loc=loc, scale=scale)) for p in thetas] 
    prior = prior/sum(prior)
    return prior    
    
    
# --------------------------------------------------------------------------------- #    
    
# These two functions are used to generate the sample X. 
skew_original = -6    
loc_original = 99    
scale_original = 27   

# Function to sample discrete random values from a skewed gaussian distribution
def randn_skew(n_samples, skew=0.0, loc=0.0, scale=1.0):
    probs = skew_gauss(skew, loc, scale)
    samples = np.random.choice(thetas, size=n_samples, replace=True, p=probs)   
    samples = list(samples)  #convert ndarray to a python list
    return samples

# Function to draw samples X for the given number of trials
# prior gives the total lifespan, but x (samples) are current ages observed
# so x should be <= total lifespan
def draw(n_trials, n_samples):
    x_vector = []
    for i in np.arange(n_trials):
        # generating Z from alpha
        z_vector = randn_skew(n_samples=n_samples, skew=skew_original, loc=loc_original, scale=scale_original)  
        x_vector.append(np.asarray([np.random.randint(low=1, high=th+1) for th in z_vector]))   # X from Z
    return x_vector 
    

# Function to compute the likelihood of each  
# 'x' in the Sample X (x_vector)
def compute_lik(x_vector):
    lik = np.zeros((len(x_vector), max_age))
    i = 0
    for obs in x_vector:
        lik[i,:] = likelihood(obs)    #pXZ
        i = i+1
    return lik     
    
# --------------------------------------------------------------------------------- #    
data = pickle.load(open(fname_input, 'rb'))
prior = data['prior']

pZA = prior
logpZA = np.log(pZA)
M = len(prior)

# --------------------------------------------------------------------------------- #    

## MODEL
for index in range(num_iterations):
  
  def prior_space():
    idx = np.random.randint(0, len(pZA))
    return pZA[idx]
  
  n_basis = 20
  
  space = nengo.FunctionSpace(
        nengo.dists.Function(prior_space),
        n_basis=n_basis)
  
  from copy import deepcopy
  space_raw = deepcopy(space.space)
  
  ### ----------------------------------------------------------------------------------------------- #    
  
  # initial input
  global p, i, k, spy
  i = 0
  k = 0
  srch_space = M

  beta = 0.65
  dt = 0.001
  
  X = draw(1, training_size+5)
  _pXZ = compute_lik(X[0])
  p = _pXZ[0]
  
  a =  np.random.randint(0, M) 
  L = np.zeros(M)   
  pxz = p * pZA[a, :]   # pZA changing every one second, p (likelihood) changing every 20 secs or so. 
  input = (pxz / np.sum(pxz))
  
  model = nengo.Network(label='learn prior') 
  with model:
    
    def ctx_drive(t, x):
        "t - current time in ms"
        "x - cortical state in cortex2 ensemble"
        global k, p, i, spy_x
        change_freq = 3
        
        # reconstruct the prior back to 120dim space
        # normalize reconstructed prior
        x = space.reconstruct(x)
        if sum(x) != 0:
            x = x/np.sum(x) 
        
        # swap
        if t%1<0.5:
            if t<1:
                spy_x = x
            else:    
                temp = x
                x = spy_x
                spy_x = temp
                    

        # every 'change_freq' iterations, a new sample comes in (so the likelihood changes)
        if t%change_freq == 0 and t!=0:
            i = i+1
            p = _pXZ[i]
        

        # updating every 1 second, so each iteration is 1s 
        if t%1 == 0 and t!=0:
            k = k+1         
        
        nk = np.power(float(np.floor(k)+2), -beta)
        if t<1:
            return nk*input
        
            
        # every 1 second, a new prior (from previous iteration) is used to update ctx
        # p is the likelihood and x is the prior selected from previous iteration
        pxz = p * x   
        if pxz.any():
            pxz = pxz / np.sum(pxz)
        else:
            # pxz has all zeros
            #print "pxz = zero"
            pass
         
        return nk*pxz
     
            
    def ctx_to_bg(t, x):
        if x.any():
            new_xmax = 1
            new_xmin = 0
            x_min = np.min(x)
            x_max = np.max(x)
            if x_min != x_max:
                x = (new_xmax - new_xmin)/(x_max - x_min)*(x - x_min)+ new_xmin 
            else:
                #all elements in the list are equal
                x = x/len(x)
        return x
    
    
    ### ------------------------------------------------------------------------------------------------------- ###  
    # Cortical Processing
    
    
    # node providing cortical input to the model - sample data as sensory input 
    cortex_in = nengo.Node(output=ctx_drive, size_out=max_age, size_in=space.n_basis)
    
    # Cortex representing the summary statistics of current and previous iterations
    # This is the input for wta network
    # SS-NN This is also where addition of current and previous statistics takes place
    ensemble_ctx = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=srch_space, neuron_type=nengo.LIF()) 
    for ens in ensemble_ctx.all_ensembles:
        ens.encoders = nengo.dists.Choice([[-1.]]) 
        ens.eval_points = nengo.dists.Uniform(-1, 0.0)
        ens.radius = 1
        ens.intercepts=nengo.dists.Uniform(-0.6, 1.)
              
           
    # connect sensory input to cortex
    # logpZA.T => (srch_space, max_age)
    # Divide by 10 to make it representable by neurons.
    nengo.Connection(cortex_in, ensemble_ctx.input, transform=logpZA/10.0)
    
        
    # Notes about ensemble_ctx:
    # 100 neurons work accurate with LIFRate neurons
    # However, with 100 neurons LIF neurons don't represent accurately need upto 500 LIF neurons  
    # -ve encoders work much better since the values represented by this ensemble are always negative
     
    
    ### ------------------------------------------------------------------------------------------------------- ###  
    # Winner take all network - Independant Accumulator
    # needs to be reset through inhibition before every iteration.
    
    
    # SS-NN
    wta = IA(d=srch_space, n_neurons=150, dt=dt)
    wta.x.add_neuron_input()

    # min'm 3s needed for inhibition
    def inhib(t):
        if t%1 > 0.5:
            return 2.0 
        else:
            return 0
    
    #Connecting inhibit population to error population
    inhibit = nengo.Node(inhib)
    nengo.Connection(inhibit, wta.x.neuron_input, 
                     transform=[[-2]] * wta.x.n_neurons_per_ensemble * wta.x.n_ensembles, synapse=0.01)
    
    
    ### ------------------------------------------------------------------------------------------------------- ###  
    # Prepare input for wta network
    # Input should be positive and scaled to span the entire range 
    # from 0-1 in order to space apart the competing values
    
    
    # node providing a constant bias of 1 to convert the values in 'ensemble_ctx' 
    # from -ve to +ve before wta input
    node_const = nengo.Node(output=np.ones(srch_space), size_out=srch_space)
    
    # node used for scaling the values in 'ensemble_ctx' to span entire range from 0-1
    scale_node = nengo.Node(output=ctx_to_bg, size_in=srch_space, size_out=srch_space)
    
    # node just for adding the input from memory network
    #NC node_ctx = nengo.Node(size_in=srch_space, size_out=srch_space)
    #NC nengo.Connection(ensemble_ctx.output, node_ctx, synapse=0.02)
    
    # node to add the bias before scaling
    # NC nengo.Connection(node_ctx, addbias_node)
    addbias_node = nengo.Node(size_in=srch_space, size_out=srch_space)
    nengo.Connection(node_const, addbias_node)
    nengo.Connection(ensemble_ctx.output, addbias_node)
        
    # scale the values only once the bias has been added
    nengo.Connection(addbias_node, scale_node) #SS
    nengo.Connection(scale_node, wta.input)


    
    ### ------------------------------------------------------------------------------------------------------- ###  
    

    # These connection leads to (1-n(k))u
    # store bg output to memory (u = memory)
    def mem_to_bg(t, x):
        if t<1:
            return 0
        #if t%1 > 0.5:
        #    return 0
        global k
        nk = np.power(float(np.floor(k)+2), -beta)
        return x * (1 - nk)  # this will be zero for the first 0.5 seconds since x is zero
      
    
    def gate_mem2(t, x):
        "gate 0 - gate is open"
        "gate 1 - gate is closed"
        "close the gate in first half, open in second"
        gate_input = 1
        if t%1 >= 0.5 and t%1 < 1:
            gate_input = 0
        elif t%1 >= 0 and t%1 < 0.5:
            gate_input = 1
        return gate_input 
    
    
    def gate_mem1(t, x):
        "gate 0 - gate is open"
        "gate 1 - gate is closed"
        "open the gate in first half, close in second"
        gate_input = 1
        if t%1 >= 0.5 and t%1 < 1:
            gate_input = 1
        elif t%1 >= 0 and t%1 < 0.5:
            gate_input = 0
        return gate_input
    
    
    ### ------------------------------------------------------------------------------------------------------- ###  
    # Memory Network
    
    # Notes:
    # memory nets can't have direct neurons
    # Don't change the recurrent synapse. Default value of 0.1 is the most stable
    # increasing number of neurons really helps to keep the representation stable
    
    
    # build two difference integrator units
    memory1 = InputGatedMemory(n_neurons=100, n_neurons_diff=30, dimensions=srch_space) 
    memory2 = InputGatedMemory(n_neurons=100, n_neurons_diff=30, dimensions=srch_space)
    
    
    gate_in1 = nengo.Node(output=gate_mem1, size_out=1, size_in=1)
    nengo.Connection(gate_in1, memory1.gate)
    
    gate_in2 = nengo.Node(output=gate_mem2, size_out=1, size_in=1)
    nengo.Connection(gate_in2, memory2.gate)
    
    nengo.Connection(ensemble_ctx.output, memory1.input, synapse=0.02)
    nengo.Connection(memory1.output, memory2.input, synapse=0.02)
    
    # node is needed to multiply by updated (1-nk) over time
    node_mem = nengo.Node(output=mem_to_bg, size_in=srch_space, size_out=srch_space)
    nengo.Connection(memory2.output, node_mem, synapse=0.02)  
    nengo.Connection(node_mem, ensemble_ctx.input)


    ### ------------------------------------------------------------------------------------------------------- ###  
    # Cortex updates at the end of an iteration
    # The winning prior needs to be stored n the cortex but the values
    # are too small, so we need special eval points and encoders.
    # Moreover dimensionality reduction is important here for computational efficiency
    
    
    # ensemble to store the winning prior for the current iteration
    cortex1 = nengo.Ensemble(n_neurons=200, dimensions=space.n_basis,
                         encoders=space.project(space_raw),
                         eval_points=space.project(space_raw),
                         neuron_type = nengo.LIF()
                         )  
    
    dummy_ctx = nengo.Ensemble(n_neurons=1, dimensions=max_age, neuron_type = nengo.Direct())  
    
    def project(x):
        return space.project(x)
    
    
    # pZA.T => (max_age, srch_space)
    # function is applied before the transform when both are on a connection
    nengo.Connection(wta.output, dummy_ctx, transform=pZA.T, synapse=0.02)
    nengo.Connection(dummy_ctx, cortex1, function=project)
    
    # prior is sent to the node 'cortex_in' to be processed 
    # and converted to the posterior.
    nengo.Connection(cortex1, cortex_in)
    
    ### ------------------------------------------------------------------------------------------------------- ###  
    # Probes
    
    # wta 
    wta_doutp = nengo.Probe(wta.output, synapse=0.02)
    
    # ctx 
    cortex1_p = nengo.Probe(cortex1, synapse=0.02)


  sim = nengo.Simulator(model, dt=dt)  # Create the simulator
  sim.run(training_size*3)                  # Run it for 1 second   
  
  
  # Collect data and write it to a pickle file
  data_out = {}
  data_out[index] = [sim.data[wta_doutp], sim.data[cortex1_p], space._basis, space._scale]
  
  pickleout = open("data_out/" + fname_output + "_" + str(index) + ".p", 'wb')
  pickle.dump(data_out, pickleout)
  pickleout.close()
  print("pickle complete for iteration: ", index, "in ", fname_output)
# ------------------------------------------------------------------------------------------------------- #  
  

