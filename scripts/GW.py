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
# | The code takes about 12 minutes to run on an L4 GPU [~38 dead points/second].

import blackjax
import blackjax.ns.adaptive
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from anesthetic import NestedSamples
from astropy.time import Time
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD

jax.config.update('jax_enable_x64', True) 

# | Define LIGO event data

gps = 1126259462.4
fmin = 20.0
fmax = 1024.0
H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

waveform = RippleIMRPhenomD(f_ref=20)
detectors = [H1, L1]
frequencies = H1.frequencies
duration=4
post_trigger_duration=2
epoch = duration - post_trigger_duration
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad

columns = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "phase_c", "psi", "ra", "dec"]
labels = [r"$M_c$", r"$q$", r"$s_{1z}$", r"$s_{2z}$", r"$\iota$", r"$d_L$", r"$t_c$", r"$\phi_c$", r"$\psi$", r"$\alpha$", r"$\delta$"]

# | Define the prior function
def logprior_fn(x):
    M_c, q, s1_z, s2_z, iota, d_L, t_c, phase_c, psi, ra, dec = x.T
    logprob = 0.0
    logprob += jax.scipy.stats.uniform.logpdf(M_c, 10.0, 80.0-10.0)
    logprob += jax.scipy.stats.uniform.logpdf(q, 0.125, 1.0-0.125)
    logprob += jax.scipy.stats.uniform.logpdf(s1_z, -1.0, 1.0+1.0)
    logprob += jax.scipy.stats.uniform.logpdf(s2_z, -1.0, 1.0+1.0)
    logprob += jnp.log(jnp.sin(iota)/2.0) + jnp.where(iota < 0.0, -jnp.inf, 0.0) + jnp.where(iota > jnp.pi, -jnp.inf, 0.0)
    logprob += jax.scipy.stats.beta.logpdf(d_L, 2.0, 2.0, 1.0, 2000.0-1.0)
    logprob += jax.scipy.stats.uniform.logpdf(t_c, -0.05, 0.05+0.05)
    logprob += jax.scipy.stats.uniform.logpdf(phase_c, 0.0, 2 * jnp.pi)
    logprob += jax.scipy.stats.uniform.logpdf(psi, 0.0, 2 * jnp.pi)
    logprob += jax.scipy.stats.uniform.logpdf(ra, 0.0, 2 * jnp.pi)
    logprob += jnp.log(jnp.cos(dec)/2.0) + jnp.where(dec < -jnp.pi/2.0, -jnp.inf, 0.0) + jnp.where(dec > jnp.pi/2.0, -jnp.inf, 0.0)
    return logprob

# | Define the likelihood function
@jax.jit
def loglikelihood_fn(x):
    params = dict(zip(columns, x.T))
    params["eta"] = 0.15874815
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
n_live = 1000
n_delete = 500
num_mcmc_steps = n_dims * 3

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
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key, 2)
init_keys = jax.random.split(init_key, 11)

M_c = jax.random.uniform(init_keys[0], (n_live,), minval=10.0, maxval=80.0)
q = jax.random.uniform(init_keys[1], (n_live,), minval=0.125, maxval=1.0)
s1_z = jax.random.uniform(init_keys[2], (n_live,), minval=-1.0, maxval=1.0)
s2_z = jax.random.uniform(init_keys[3], (n_live,), minval=-1.0, maxval=1.0)
iota = 2*jnp.arcsin(jax.random.uniform(init_keys[4], (n_live,))**0.5)
d_L = jax.random.beta(init_keys[5], 2.0, 2.0, (n_live,))*(2000.0-1.0)+1.0
t_c = jax.random.uniform(init_keys[6], (n_live,), minval=-0.05, maxval=0.05)
phase_c = jax.random.uniform(init_keys[7], (n_live,), minval=0.0, maxval=2 * jnp.pi)
psi = jax.random.uniform(init_keys[8], (n_live,), minval=0.0, maxval=2 * jnp.pi)
ra = jax.random.uniform(init_keys[9], (n_live,), minval=0.0, maxval=2 * jnp.pi)
dec = 2*jnp.arcsin(jax.random.uniform(init_keys[10], (n_live,))**0.5)-jnp.pi/2.0

initial_particles = jnp.vstack([M_c, q, s1_z, s2_z, iota, d_L, t_c, phase_c, psi, ra, dec]).T
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
samples.to_csv('GW.csv')
