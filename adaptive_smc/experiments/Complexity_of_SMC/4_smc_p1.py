r"""
The initial distribution is N(0, C0), the target distribution is
N(0, C1) (product of densities),
where C0, C1 are two random scaled positive covariance matrices
so that they are well-conditioned (e.g., with conditioning number independent on d)
ubt potentially the target has heavier tails than the base.

The sequence of distributions is the tempered interpolation with temperature
obtained by automatically setting ESS >= N/2.

The Markov kernels are Metropolis Hastings kernel with RWM set to 2.38^2/d.

Standard SMC with P=1 (one kernel application per particle and per iteration), with the
number of tempering steps multiplied by the P of 4_smc.py, so that the total number of
kernel applications matches 4_smc.py (referee 1, "tempered SMC samplers with P=1").

The saved output is the weights.
"""

from adaptive_smc.SMC import GenericTemperingSMC
from adaptive_smc.experiments.Complexity_of_SMC.heavy_tail_increment_weights_gaussians import make_model
from adaptive_smc.experiments.Complexity_of_SMC.proposal import \
    build_gaussian_rw_proposal_fixed_scaling
from adaptive_smc.experiments.Complexity_of_SMC.save import save

import os
import sys
from datetime import datetime
import jax
import jax.numpy as jnp
import yaml

jax.config.update("jax_enable_x64", False)

OP_key = jax.random.PRNGKey(12)
_, key = jax.random.split(OP_key)


def default_title(prefix='', eps=None):
    now = datetime.now()
    filename = (
        f"{prefix}_eps{eps:.4f}_"
        f"{os.path.basename(__file__)}_"
        f"{now.strftime('%m%d%H%M%S')}.pkl"
    )
    return filename


def xp(config, eps, keys):
    # Copy config so we never mutate the original
    run_config = dict(config)

    dim = run_config["dim"]
    target_ess = run_config.get("ess", None)

    T = int(jnp.sqrt(dim))
    gamma = 1 / dim

    num_parallel_chain = 20 * dim
    budget_num_mcmc_steps = int(
        run_config["num_mcmc_steps"] * dim ** 1.0 // 2)  # P of 4_smc.py
    num_mcmc_steps = 1
    tempering_length = T * 2 * budget_num_mcmc_steps

    tempering_sequence = jnp.linspace(0, 1, tempering_length)

    # Generate model
    model_key = jax.random.PRNGKey(0)
    loglikelihood_fn, base_measure_sampler, logbase_density_fn, true_log_Z, _, _ = make_model(dim,
                                                                                        config.get('heavy_factor', 1.0),
                                                                                        model_key,
                                                                                       config.get('mean', None))

    run_config.update({
        "eps": eps,
        "P": num_mcmc_steps,
        "T": tempering_length,
        "M": num_parallel_chain,
        "logZ": true_log_Z,
        "tempering_sequence": tempering_sequence
    })

    smc = GenericTemperingSMC(
        logbase_density_fn,
        base_measure_sampler,
        loglikelihood_fn,
        build_gaussian_rw_proposal_fixed_scaling
    )

    @jax.jit
    @jax.vmap
    def wrapper_smc(key):
        return smc.low_memory_sample(
            key,
            num_parallel_chain,
            num_mcmc_steps,
            tempering_sequence,
            target_ess
        )

    res = wrapper_smc(keys)

    output_file = (
            run_config["OUTPUT_PATH"]
            + default_title(run_config.get("prefix", ""), eps)
    )

    save(res, run_config, output_file)


def plot_zhat_boxplots(dirs, output_file=None, title=None):
    r"""
    Boxplots of \hat{Z} / Z for each dimension d, one box per sampler.

    dirs: dict {label: directory}, each directory holding the "__*.pkl" outputs of
    4_smc.py or 4_smc_p1.py. \hat{Z} is normalised by the true Z so that all dimensions
    share the same axis; each box is annotated with the P and T of the corresponding runs.
    """
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    # data[label][dim] = (ratios Zhat / Z, P, T)
    data = {}
    for label, directory in dirs.items():
        data[label] = {}
        if not os.path.isdir(directory):
            print(f"plot_zhat_boxplots: no directory {directory}, skipping {label}")
            continue
        for file in sorted(os.listdir(directory)):
            if not (file.endswith(".pkl") and file.startswith("__")):
                continue
            with open(os.path.join(directory, file), "rb") as f:
                my_pkl = pickle.load(f)
            config = my_pkl["config"]
            log_ratios = np.asarray(my_pkl["res"][4]).sum(-1) - float(config["logZ"])
            T = len(np.asarray(config["tempering_sequence"]))
            ratios = data[label].get(config["dim"], (np.empty(0),))[0]
            data[label][config["dim"]] = (np.concatenate([ratios, np.exp(log_ratios)]), config["P"], T)

    dims = sorted({dim for per_dim in data.values() for dim in per_dim})
    labels = list(dirs)
    width = 0.8 / len(labels)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(1.5 * len(dims) + 4, 4.5))
    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    for j, label in enumerate(labels):
        present = [dim for dim in dims if dim in data[label]]
        if not present:
            print(f"plot_zhat_boxplots: no results for {label}")
            continue
        positions = [dims.index(dim) + (j - (len(labels) - 1) / 2) * width for dim in present]
        ax.boxplot(
            [data[label][dim][0] for dim in present],
            positions=positions,
            widths=0.9 * width,
            patch_artist=True,
            boxprops=dict(facecolor=colors[j], alpha=0.6),
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.3),
        )
        for pos, dim in zip(positions, present):
            _, P, T = data[label][dim]
            ax.text(pos, -0.02, f"P={P}\nT={T}", transform=text_transform, clip_on=False,
                    ha="center", va="top", fontsize=7, color=colors[j])
        ax.plot([], [], color=colors[j], linewidth=8, alpha=0.6, label=label)  # legend handle

    ax.axhline(1., color="black", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"$d={dim}$" for dim in dims])
    ax.tick_params(axis="x", length=0, pad=26)  # leave room for the P, T annotations
    ax.set_xlabel("Dimension")
    ax.set_ylabel(r"$\hat{Z} / Z$")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.), frameon=True, facecolor="white", edgecolor="0.3")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300)
    return fig


if __name__ == "__main__":

    yaml_file = "4_smc_p1.yaml"

    with open(yaml_file, "r") as file:
        y_config = yaml.safe_load(file)

    # python 4_smc_p1.py plot: compare with the outputs of 4_smc.py (./power2/ instead of ./power2P1/)
    if sys.argv[1:] == ["plot"]:
        for name_of_my_config, config in y_config.items():
            output_path = config["OUTPUT_PATH"]
            plot_zhat_boxplots(
                {"SMC (4_smc.py)": output_path.replace("power2P1", "power2"), "SMC, $P=1$": output_path},
                output_file=output_path + "zhat_boxplots.pdf",
                title=f"{name_of_my_config}, heavy_factor = {config.get('heavy_factor', 1.0)}",
            )
        sys.exit()

    for name_of_my_config, config in y_config.items():

        if config.get("run", True):

            # Fix epsilon for this scaling experiment
            eps = config.get("eps", 0.1)

            # Set a list of dimensions to vary
            dim_list = config.get("dim_list", [5])

            sequential_repetitions = config.pop("sequential_repetitions", 1)
            parallel_repetitions = config["parallel_repetitions"]

            seq_keys = jax.random.split(key, sequential_repetitions)

            all_keys = jax.vmap(
                lambda k: jax.random.split(k, parallel_repetitions)
            )(seq_keys)

            _, key = jax.random.split(seq_keys[-1])

            # Loop over dimensions
            for dim in dim_list:
                # update config with current dim
                config["dim"] = dim

                for keys in all_keys:
                    xp(config, eps, keys)