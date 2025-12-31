# ML Workflow Template

## Installation

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
uv venv
```

Activate the virtual environment

Download dependencies

```bash
uv sync
```

The entry point is `train`, to start training use:

```bash
train
```

## Shell Completions

**Bash - Install**:

```bash
eval "$(train -sc install=bash)"
```

**Bash - Uninstall**:

```bash
eval "$(train -sc uninstall=bash)"
```

## Configure an Experiment

An experiment can be declared in `src/config/experiment`.
To run the experiment use `+experiment=` syntax.

```bash
train +experiement=mnist_cnn
```

> [!NOTE]
> You can still perform overriding of the hyperparameters

## Use Grid Search

Grid search can be declared in `src/config/grid`.

The default sweeper must be configured to specify the serach space:

```yaml
defaults:
  - override /data: mnist
  - override /model: classifier
  - override /model/net: cnn
  - override /optim: sgd

hydra:
  mode: "MULTIRUN" # set hydra to multirun by default if this config is attached

  sweeper:

    # define hyperparameter search space
    params:
      data.batch_size: 16, 32
      optim.lr: 1e-1
```

The configuration above will run 2 jobs.

To run a grid search use `+grid=` syntax

```bash
train +grid=mnist_cnn
```

## Search hperparameters with Opuna

The search with Bayesian Optimization can be configured in `src/config/hparam_search`.

We use optuna sweeper (from the following [PR](https://github.com/facebookresearch/hydra/pull/3046)
to use newer version of optuna)

A metric to be optimized must be specified. It is recommendad to specify the metric once,
like in the example below the `optimized_metric` is read from the main file under `loss.monitor`.

We can also use a `db` to store the study and later querying it or using optuna dashboard to get plots and insights.

```yaml
# @package _global_

# example hyperparameter optimization with Optuna:
# train +hparams_search=mnist_cnn

defaults:
  - override /hydra/sweeper: optuna
  - override /data: mnist
  - override /optim: adam
  - override /model: classifier
  - override /model/net: cnn

optimized_metric: ${loss.monitor}

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
# docs: https://hydra.cc/docs/next/plugins/optuna_sweeper
hydra:
  mode: "MULTIRUN" # set hydra to multirun by default if this config is attached

  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # storage URL to persist optimization results
    # for example, you can use SQLite if you set 'sqlite:///example.db'
    storage: "sqlite:///mnist.db"

    # name of the study to persist optimization results
    study_name: "mnist_cnn"

    # number of parallel workers
    n_jobs: 1

    # 'minimize' or 'maximize' the objective
    direction: 'minimize'

    # total number of runs that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # you can choose bayesian sampler (tpe), other samplers don't work
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: ${seed}
      n_startup_trials: 10 # number of random sampling runs before optimization starts

    # define hyperparameter search space
    params:
      data.batch_size: choice(1, 16, 32)
      optim.lr: range(1e-4, 1e-3)
```

> [!WARNING]
> Pruning is not availabel with the hydra-sweeper