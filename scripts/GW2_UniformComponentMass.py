# | # Minimal example of GW parameter estimation
# | 
# | This script performs Bayesian inference on LIGO data (from GW150914) using
# | a nested sampling algorithm implemented with BlackJAX. It loads the 
# | detector data and sets up a gravitational-wave waveform model, defines a 
# | prior and likelihood for the model parameters, then runs nested sampling 
# | to sample from the posterior. Finally, it processes the samples with 
# | anesthetic and writes them to a CSV file.
# |
# | ## Installation
# |```bash
# | python -m venv venv
# | source venv/bin/activate
# | pip install git+https://git.ligo.org/lscsoft/ligo-segments.git
# | pip install git+https://github.com/kazewong/jim
# | pip install git+https://github.com/handley-lab/blackjax@proposal
# | pip install anesthetic
# | python GW.py
# |```
# | The code takes about 68 minutes to run on an L4 GPU
# | [nlive=3000, n_delete=1000, mcmc=5*dim, ~21.6 dead points/second].

import blackjax
import blackjax.ns.adaptive
import jax
import jax.scipy.stats as stats
import jax.numpy as jnp
import numpy as np
import tqdm
from anesthetic import NestedSamples
from astropy.time import Time
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD

# Define parameters class
class ParameterPrior:
    """
    Class to define a parameter with its name, label, and prior distribution.

    Parameters:
    name: str
        The name of the parameter.
    label: str
        The label of the parameter.
    prior_fn: callable function
        The prior distribution function.
    args: tuple
        The arguments of the prior distribution function.
    """
    def __init__(self, name: str, label: str, prior_fn: callable, *args):
        self.name = name
        self.label = label
        self.prior_fn = prior_fn
        self.args = args

    def logprob(self, value: float) -> float:
        """
        Calculates the log probability of the given parameter value by calling
        the defined log-prior function.
        """
        return self.prior_fn(value, *self.args)

# Define the prior functions
def logUniformPrior(x: float, min: float, max: float) -> float:
    """
    Log-uniform prior distribution.

    Args:
    x: float
        The parameter value.
    min: float
        The minimum value of the parameter.
    max: float
        The maximum value of the parameter.
    """
    
    return stats.uniform.logpdf(x, min, max-min)

def logSinPrior(x):
    """
    Log-sin prior distribution. Boundaries are at 0 and pi.

    Args:
    x: float
        The parameter value.
    """
    return jnp.log(jnp.sin(x)/2.0) + jnp.where(x < 0.0, -jnp.inf, 0.0) + jnp.where(x > jnp.pi, -jnp.inf, 0.0)

def logCosPrior(x):
    """
    Log-cos prior distribution. Boundaries are at -pi/2 and pi/2.

    Args:
    x: float
        The parameter value.
    """
    return jnp.log(jnp.cos(x)/2.0) + jnp.where(x < -jnp.pi/2.0, -jnp.inf, 0.0) + jnp.where(x > jnp.pi/2.0, -jnp.inf, 0.0)

def logBetaPrior(x, min, max):
    """
    Log-Volumetric prior for luminosity distance. 

    Args:
    x: float
        The parameter value.
    min: float
        The minimum value of the parameter.
    max: float
        The maximum value of the parameter.
    """
    return stats.beta.logpdf(x, 3.0, 1.0, min, max-min)


jax.config.update('jax_enable_x64', True) 
label = 'GW2_CompMass'

# | Define LIGO event data
gps = 1126259462.4
fmin = 20.0
fmax = 1024.0
duration = 8
post_trigger_duration = 2
end_time = gps + post_trigger_duration
start_time = end_time - duration
roll_off = 0.4
tukey_alpha = 2 * roll_off / duration
psd_pad = 16

detectors = [H1, L1]
H1.load_data(gps, duration - post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=psd_pad, tukey_alpha=tukey_alpha)
L1.load_data(gps, duration - post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=psd_pad, tukey_alpha=tukey_alpha)

waveform = RippleIMRPhenomD(f_ref=fmin)
frequencies = H1.frequencies
epoch = duration - post_trigger_duration
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

# Define the priors.
parameters = [
    ParameterPrior("M_1", r"$M_1$", logUniformPrior, 10.0, 80.0),
    ParameterPrior("M_2", r"$M_2$", logUniformPrior, 10.0, 80.0), 
    ParameterPrior("s1_z", r"$s_{1z}$", logUniformPrior, -1.0, 1.0),
    ParameterPrior("s2_z", r"$s_{2z}$", logUniformPrior, -1.0, 1.0),
    ParameterPrior("iota", r"$\iota$", logSinPrior),
    ParameterPrior("d_L", r"$d_L$", logBetaPrior, 1.0, 2000.0),
    ParameterPrior("t_c", r"$t_c$", logUniformPrior, -0.05, 0.05),
    ParameterPrior("phase_c", r"$\phi_c$", logUniformPrior, 0.0, 2 * jnp.pi),
    ParameterPrior("psi", r"$\psi$", logUniformPrior, 0.0, jnp.pi),
    ParameterPrior("ra", r"$\alpha$", logUniformPrior, 0.0, 2 * jnp.pi),
    ParameterPrior("dec", r"$\delta$", logCosPrior)  
]

columns = [param.name for param in parameters]
labels = [param.label for param in parameters]

# | Define the log prior function
def logprior_fn(x):
    logprob = 0.0
    for param, value in zip(parameters, x.T):
        logprob += param.logprob(value)
    return logprob

# | Define the likelihood function
@jax.jit
def loglikelihood_fn(x):
    params = dict(zip([param.name for param in parameters], x.T))
    params["M_c"] = (params["M_1"]*params["M_2"])**0.6 / (params["M_1"]+params["M_2"])**0.2
    params["eta"] = params["M_1"]*params["M_2"] / (params["M_1"]+params["M_2"])**2
    params["eta"] = jax.lax.cond(
        jnp.isclose(params["eta"], 0.25),
        lambda _: 0.249995,
        lambda _: params["eta"],
        operand=None
    )
    
    params["gmst"] = gmst
    waveform_sky = waveform(frequencies, params)
    align_time = jnp.exp(-1j * 2 * jnp.pi * frequencies * (epoch + params["t_c"]))
    return likelihood_function(
        params,
        waveform_sky,
        detectors,
        frequencies,
        align_time,
    )

# | Define the Nested Sampling algorithm
n_dims = len(columns)
n_live = 2000
n_delete = 800
num_mcmc_steps = n_dims * 5

# | Initialize the Nested Sampling algorithm
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
)

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

# | Sample live points from the prior
# Check: Is match -> case a more efficient way of doing this or is the diff. negligible?
def sample_prior(parameter, key, n_live):
    if parameter.prior_fn == logUniformPrior:
        return jax.random.uniform(key, (n_live,), minval=parameter.args[0], maxval=parameter.args[1])
    elif parameter.prior_fn == logSinPrior:
        return 2 * jnp.arcsin(jax.random.uniform(key, (n_live,)) ** 0.5)
    elif parameter.prior_fn == logCosPrior:
        return 2 * jnp.arcsin(jax.random.uniform(key, (n_live,)) ** 0.5) - jnp.pi / 2.0
    elif parameter.prior_fn == logBetaPrior:
        return jax.random.beta(key, 3.0, 1.0, (n_live,)) * (parameter.args[1] - parameter.args[0]) + parameter.args[0]

rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key, 2)
init_keys = jax.random.split(init_key, len(parameters))

initial_particles = jnp.vstack([sample_prior(param, key, n_live) for param, key in zip(parameters, init_keys)]).T
# q_index = columns.index("q")
# initial_particles = initial_particles.at[:, q_index].set(
#    initial_particles[:, q_index] / (1 + initial_particles[:, q_index]) ** 2
#)
state = nested_sampler.init(initial_particles, loglikelihood_fn)

# | Run Nested Sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)  # Update progress bar

# | anesthetic post-processing
dead = jax.tree.map(
        lambda *args: jnp.reshape(jnp.stack(args, axis=0), 
                                  (-1,) + args[0].shape[1:]),
        *dead)
live = state.sampler_state
logL = np.concatenate((dead.logL, live.logL), dtype=float)
logL_birth = np.concatenate((dead.logL_birth, live.logL_birth), dtype=float)
data = np.concatenate((dead.particles, live.particles), dtype=float)
samples = NestedSamples(data, logL=logL, logL_birth=logL_birth, columns=columns, labels=labels)
samples.to_csv(f'{label}.csv')

logzs = samples.logZ(100)
print(f"{logzs.mean()} +- {logzs.std()}")