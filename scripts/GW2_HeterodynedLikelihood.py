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
from jimgw.single_event.detector import Detector, H1, L1
from jimgw.single_event.likelihood import original_relative_binning_likelihood as relative_binning_likelihood_function
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
        return self.prior_fn(value, *args)

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
label = 'GW2'

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
    ParameterPrior("M_c", r"$M_c$", logUniformPrior, 10.0, 80.0),
    ParameterPrior("q", r"$q$", logUniformPrior, 0.125, 1.0), 
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



# | SETUP HETERODYNING
# |
# |
# |
from jaxtyping import Array, Float
import numpy.typing as npt
from scipy.interpolate import interp1d

def max_phase_diff(
    f: npt.NDArray[np.floating],
    f_low: float,
    f_high: float,
    chi: Float = 1.0,
    ):
    """
    Compute the maximum phase difference between the frequencies in the array.

    Parameters
    ----------
    f: Float[Array, "n_dim"]
        Array of frequencies to be binned.
    f_low: float
        Lower frequency bound.
    f_high: float
        Upper frequency bound.
    chi: float
        Power law index.

    Returns
    -------
    Float[Array, "n_dim"]
        Maximum phase difference between the frequencies in the array.
    """

    gamma = np.arange(-5, 6, 1) / 3.0
    f = np.repeat(f[:, None], len(gamma), axis=1)
    f_star = np.repeat(f_low, len(gamma))
    f_star[gamma >= 0] = f_high
    return 2 * np.pi * chi * np.sum((f / f_star) ** gamma * np.sign(gamma), axis=1)

def compute_coefficients(data, h_ref, psd, freqs, f_bins, f_bins_center):
    A0_array = []
    A1_array = []
    B0_array = []
    B1_array = []

    df = freqs[1] - freqs[0]
    data_prod = np.array(data * h_ref.conj())
    self_prod = np.array(h_ref * h_ref.conj())
    for i in range(len(f_bins) - 1):
        f_index = np.where((freqs >= f_bins[i]) & (freqs < f_bins[i + 1]))[0]
        A0_array.append(4 * np.sum(data_prod[f_index] / psd[f_index]) * df)
        A1_array.append(
            4
            * np.sum(
                data_prod[f_index]
                / psd[f_index]
                * (freqs[f_index] - f_bins_center[i])
            )
            * df
        )
        B0_array.append(4 * np.sum(self_prod[f_index] / psd[f_index]) * df)
        B1_array.append(
            4
            * np.sum(
                self_prod[f_index]
                / psd[f_index]
                * (freqs[f_index] - f_bins_center[i])
            )
            * df
        )

    A0_array = jnp.array(A0_array)
    A1_array = jnp.array(A1_array)
    B0_array = jnp.array(B0_array)
    B1_array = jnp.array(B1_array)
    return A0_array, A1_array, B0_array, B1_array

class HeterodynedLikelihood():
    def __init__(self, detectors: list[Detector], waveform, frequencies, epoch, gmst):
        self.detectors = detectors
        self.waveform = waveform
        self.frequencies = frequencies
        self.epoch = epoch
        self.gmst = gmst
        self.n_bins = 100
        self.A0_array = {}
        self.A1_array = {}
        self.B0_array = {}
        self.B1_array = {}
        self.waveform_low_ref = {}
        self.waveform_center_ref = {}

    def make_binning_scheme(
        self, freqs: npt.NDArray[np.floating], n_bins: int, chi: float = 1
    ) -> tuple[Float[Array, " n_bins+1"], Float[Array, " n_bins"]]:
        """
        Make a binning scheme based on the maximum phase difference between the
        frequencies in the array.

        Parameters
        ----------
        freqs: Float[Array, "dim"]
            Array of frequencies to be binned.
        n_bins: int
            Number of bins to be used.
        chi: float = 1
            The chi parameter used in the phase difference calculation.

        Returns
        -------
        f_bins: Float[Array, "n_bins+1"]
            The bin edges.
        f_bins_center: Float[Array, "n_bins"]
            The bin centers.
        """

        phase_diff_array = max_phase_diff(freqs, freqs[0], freqs[-1], chi=chi)
        bin_f = interp1d(phase_diff_array, freqs)
        f_bins = np.array([])
        for i in np.linspace(phase_diff_array[0], phase_diff_array[-1], n_bins + 1):
            f_bins = np.append(f_bins, bin_f(i))
        f_bins_center = (f_bins[:-1] + f_bins[1:]) / 2
        return jnp.array(f_bins), jnp.array(f_bins_center)
        
    def reference_state(self, params):
        h_sky = waveform(self.frequencies, params)
        # Get the grid of the relative binning scheme (contains the final endpoint)
        # and the center points
        freq_grid, self.freq_grid_center = self.make_binning_scheme(
            np.array(self.frequencies), self.n_bins
            )
        self.freq_grid_low = freq_grid[:-1]
        params["gmst"] = self.gmst
        if jnp.isclose(params["eta"], 0.25):
            params["eta"] = 0.249995
        # Get frequency masks to be applied, for both original
        # and heterodyne frequency grid
        h_amp = jnp.sum(
            jnp.array([jnp.abs(h_sky[key]) for key in h_sky.keys()]), axis=0
        )
        f_valid = self.frequencies[jnp.where(h_amp > 0)[0]]
        f_max = jnp.max(f_valid)
        f_min = jnp.min(f_valid)

        mask_heterodyne_grid = jnp.where((freq_grid <= f_max) & (freq_grid >= f_min))[0]
        mask_heterodyne_low = jnp.where(
            (self.freq_grid_low <= f_max) & (self.freq_grid_low >= f_min)
        )[0]
        mask_heterodyne_center = jnp.where(
            (self.freq_grid_center <= f_max) & (self.freq_grid_center >= f_min)
        )[0]
        freq_grid = freq_grid[mask_heterodyne_grid]
        self.freq_grid_low = self.freq_grid_low[mask_heterodyne_low]
        self.freq_grid_center = self.freq_grid_center[mask_heterodyne_center]

        # Assure frequency grids have same length
        if len(self.freq_grid_low) > len(self.freq_grid_center):
            self.freq_grid_low = self.freq_grid_low[: len(self.freq_grid_center)]

        h_sky_low = self.waveform(self.freq_grid_low, params)
        h_sky_center = self.waveform(self.freq_grid_center, params)
        # Get phase shifts to align time of coalescence
        align_time = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.frequencies
            * (self.epoch + params["t_c"])
        )
        align_time_low = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.freq_grid_low
            * (self.epoch + params["t_c"])
        )
        align_time_center = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.freq_grid_center
            * (self.epoch + params["t_c"])
        )

        for detector in self.detectors:
            waveform_ref = (
                detector.fd_response(self.frequencies, h_sky, params)
                * align_time
            )
            self.waveform_low_ref[detector.name] = (
                detector.fd_response(self.freq_grid_low, h_sky_low, params)
                * align_time_low
            )
            self.waveform_center_ref[detector.name] = (
                detector.fd_response(
                    self.freq_grid_center, h_sky_center, params
                )
                * align_time_center
            )
            A0, A1, B0, B1 = compute_coefficients(
                detector.data,
                waveform_ref,
                detector.psd,
                self.frequencies,
                freq_grid,
                self.freq_grid_center,
            )
            self.A0_array[detector.name] = A0[mask_heterodyne_center]
            self.A1_array[detector.name] = A1[mask_heterodyne_center]
            self.B0_array[detector.name] = B0[mask_heterodyne_center]
            self.B1_array[detector.name] = B1[mask_heterodyne_center]

    def evaluate(self, params: dict[str, Float]) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        params["gmst"] = self.gmst
        # evaluate the waveforms as usual
        waveform_sky_low = self.waveform(frequencies_low, params)
        waveform_sky_center = self.waveform(frequencies_center, params)
        align_time_low = jnp.exp(
            -1j * 2 * jnp.pi * frequencies_low * (self.epoch + params["t_c"])
        )
        align_time_center = jnp.exp(
            -1j * 2 * jnp.pi * frequencies_center * (self.epoch + params["t_c"])
        )
        return relative_binning_likelihood_function(
            params,
            self.A0_array,
            self.A1_array,
            self.B0_array,
            self.B1_array,
            waveform_sky_low,
            waveform_sky_center,
            self.waveform_low_ref,
            self.waveform_center_ref,
            self.detectors,
            frequencies_low,
            frequencies_center,
            align_time_low,
            align_time_center
        )
# |
# |
# |

likelihood_function = HeterodynedLikelihood(detectors, waveform, frequencies, epoch, gmst)

# | Define the likelihood function
@jax.jit
def loglikelihood_fn(x):
    params = dict(zip([param.name for param in parameters], x.T))
    params["eta"] = params["q"] / (1 + params["q"]) ** 2
    params["eta"] = jax.lax.cond(
        jnp.isclose(params["eta"], 0.25),
        lambda _: 0.249995,
        lambda _: params["eta"],
        operand=None
    )
    return likelihood_function.evaluate(params)

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
state = nested_sampler.init(initial_particles, loglikelihood_fn)
likelihood_function.reference_state(initial_particles)

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