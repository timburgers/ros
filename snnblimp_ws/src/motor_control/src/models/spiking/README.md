# Spiking

Collection of spiking neurons (in [JAX](https://github.com/google/jax)/[flax](https://github.com/google/flax) and PyTorch) and utilities with the aim of standardizing these across papers/repositories.

## Installation

### Use as Git submodule

Follow these steps if you want to include this repository in your project:

1. Determine where you want to include your spiking components (e.g., `models/spiking`)
2. Add as submodule: `git submodule add git@github.com:tudelft/spiking.git models/spiking`
3. To pull new changes, use `git submodule update` (or go into `models/spiking`, and pull as usual)

If you clone a repository that already has submodules, make sure you either do
- `git clone ... --recurse-submodules`
- or run `git submodules init && git submodules update` afterwards.

### Development

1. Clone the repository
2. Set up a virtual environment of your choice and install requirements (see below)
4. Install pre-commit hooks: `pre-commit install`
5. Get to work!

In order to have JAX and PyTorch play nice with each other and with your drivers, install with conda (get miniconda [here](https://docs.conda.io/en/latest/miniconda.html)):

```bash
conda env create -f env.yaml
conda activate jax
```

This includes all requirements for the base spiking library. To run tests, also do `pip install -r requirements-test.txt`.

## Contents

Until we decide on JAX or PyTorch (keuzestress), and because it is nice to benchmark the two, there's a JAX/flax and a PyTorch part.

### JAX

Currently, there are two different implementations of neurons in JAX/flax (again keuzestress), and they differ in how they keep track of state:
- Regular: you initialize state in the right shape outside, and feed it to the model/layer/neuron
- Variables: you register variables that keep track of state where they are needed giving you
    1. nice organization
    2. shape inference
    3. a boost in performance (in some cases, and also disputed [here](https://github.com/google/flax/discussions/1633))

### PyTorch

#### Neurons

The base classes try to incorporate all elements of a neuron in an abstracted manner, such that these can be reused between neuron models. For instance:

- Activation is specified per neuron
- State is reset based on the input vector shape
- Parameters are fed as dicts of tensors
    - Proper shape is given outside
    - Any parameter can be either fixed or learnable
- Separate methods for updates to current, voltage, etc., such that these can be substituted

#### Layers

For the layers, we want to allow both feedforward and recurrent and both linear and convolutional, in combination with any neuron model. There are multiple ways to do this:

- Default (I like medium):
    - Specify a base class for each type of connection
    - Provide the neuron as a class variable and instantiate in init
- v2 (I like):
    - Specify a base class for each type of connection
    - Overwrite the neuron's `.forward()`
    - Let the neuron layer inherit from both this base as well as the neuron class
    - However, this gives problems somewhere I think (don't remember where)
- v3 (not recommended):
    - Write a separate class for each combination of connection and neuron

#### Surrogate gradients

We have included many types of surrogate gradients, and have made it easy to add new ones yourself.

## Discussions

### JAX? Flax?

As the developers [explain](github.com/google/jax) it, JAX is three things:

1. Autograd
2. XLA
3. Pure function transformations

[Autograd](https://github.com/HIPS/autograd) is an older library for differentiating any Python or NumPy function, so maximum flexibility without changing a lot of code. It's not actively being developed anymore, and the developers went to the JAX team.

[XLA](https://www.tensorflow.org/xla) is a compilation library (just-in-time compilation) that TensorFlow can also use (but doesn't by default). Actually, PyTorch can also use [XLA](https://github.com/pytorch/xla). It allows you to compile your Python and NumPy code and run it on GPU. XLA can also call cuDNN kernels when running on NVIDIA GPUs

JAX by itself is purely functional, and JAX arrays (wrappers around NumPy arrays) are immutable. In order to make classes you have to use a higher-level extension like [flax](github.com/google/flax), which is what is used here, but there's also [haiku](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) from DeepMind.

### JAX or PyTorch?

Why use JAX over PyTorch?
- Has both backward as well as forward-mode differentiation, which, would allow for gradient-based online learning
    - PyTorch is working on this, see [here](https://github.com/pytorch/pytorch/issues/10223), but it doesn't work yet for a lot of ops
- JAX is lower-level than PyTorch, and hence I think it allows easier integration of learning that doesn't involve gradients; like Hebbian learning
    - However, [this paper](https://arxiv.org/abs/2107.01729) (Hebbian learning) and [this paper](http://arxiv.org/abs/2102.00428) show that PyTorch is also capable of this
- There are no spiking libraries in JAX, except for [Rockpool](https://github.com/SynSense/rockpool) (SynSense), which has both JAX and PyTorch, but feels very bloated, and seems to have a lot in common with flax (which is more common)

Why use PyTorch over JAX?
- JAX has a much smaller userbase than PyTorch, so it can be more difficult to find answers
    - Luckily the developers of JAX and flax are very responsive on their respective [GitHub](https://github.com/google/jax/discussions) [discussions](https://github.com/google/flax/discussions)
    - JAX is newer than PyTorch and hasn't been optimized as much, which is both good (there is still speed to gain) and bad (there might be cases where PyTorch is faster)
    - JAX is definitely less user-friendly than PyTorch, it has many [sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html), but this will get better as we become more familiar
    - I still have to check the compatibility of convolutions and forward-mode diff and custom gradients with sparse tensors, which would make a lot of sense to use with SNNs
        - But it's not like these issues aren't there with PyTorch; adoption of sparse tensors is still limited

## Tests

### Loihi benchmark - INRC Cloud

Obtain an account for the INRC Cloud and login over SSH with a config like this:

```ssh
# INRC cloud proxy
Host ssh.intel-research.net
User=<username>
IdentityFile ~/.ssh/vlab_gateway_rsa

# Actual INRC cloud
Host ncl-edu.research.intel-research.net
HostName %h
User=<username>
ProxyCommand=ssh -W %h:%p ssh.intel-research.net
IdentityFile ~/.ssh/vlab_ext_rsa
```

Once logged in, follow [these](https://intel-ncl.atlassian.net/wiki/spaces/NAP/pages/4456463/Installation+Guide) install instructions to set up NxSDK etc. Some notes:

- Python version 3.8.10
- Ran `pip -U pip setuptools wheel`
- NxSDK version 1.0.0

To clone this repository, set up an SSH key to work with Github:
```bash
ssh-keygen -t ed25519 -f .ssh/github -C "inrc access to github"
```

Make sure to add the public part to your Github account, and add this to your SSH config on the INRC Cloud:
```ssh
Host github.com
User=git
IdentityFile ~/.ssh/github
```

Now you can clone the repo via `git clone git@github.com:tudelft/spiking.git`. Install both requirements files in the virtual environment created before. To make things easier, install the VSCode Python extension remotely and specify the location of the venv.

Plotting figures/X-forwarding doesn't work via VSCode, so manually connect over SSH if you want that:
```bash
ssh -X -Y <username>@ncl-edu.research.intel-research.net
```

Run with Loihi as follows:
```bash
SLURM=1 python tests/loihi.py --loihi
```

### JAX comparisons

#### JAX vs PyTorch - MNIST
We have a comparison for JAX/flax and PyTorch on MNIST with a convolutional SNN. PyTorch is used for data loading (also with JAX), and Tensorboard can be used for logging. With some modifications, JAX is faster! The following commands allow running the various files (note the environment variables):
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true python tests/mnist_jax.py
python tests/mnist_torch.py
```

The dataloading script compares different kinds of data loading, which turned out to be the main culprit for bad performance with JAX:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python tests/dataloading.py
```

There are still some peculiarities, see the TODOs.

Lessons learned:
- Both [JAX](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) and [TensorFlow](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth) preallocate almost all GPU memory (relevant [issue](https://github.com/google/jax/issues/8362)); to prevent this you need to set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `TF_FORCE_GPU_ALLOW_GROWTH=true`
- Currently, to have multiple PyTorch dataloader workers work with JAX/XLA, you need to set the start method to "spawn" (see [here](https://github.com/google/jax/issues/3382))
- The TensorBoard profiler should [work](https://jax.readthedocs.io/en/latest/profiling.html) with JAX, however no luck so far
- With good settings, it seems that JAX/Flax is competitive with PyTorch
- Settings:
    - JIT train and eval steps
    - Load data using Numpy, not JAX (VERY big difference)
    - Use PyTorch dataloaders with:
        - ~4 workers per GPU
        - "spawn" start method
        - persistent workers (VERY big difference for "spawn")
        - not pinning memory (no observed benefit)
    - State/carry either fed or as variable
- JAX has relatively high dispatch cost, so only becomes competitive with/better than PyTorch for larger matrices; see [here](https://github.com/google/jax/discussions/8497)

#### JAX vs PyTorch - sequential MNIST

To demonstrate BPTT and RTRL/its approximations well, we need a sequential problem, so this is that. For now, it's only BPTT though. The performance differences between PyTorch and JAX are huge! Of course, JAX is jitted, but still.

- `smnist_torch.py`: PyTorch implementation, 40s per epoch
- `smnist_jax.py`: JAX implementation with flax lifted scan, state as input, 8s per epoch
- `smnist_jax2.py`: JAX implementation with jax scan, state as variable, 7s per epoch

#### JAX vs PyTorch/Norse - influence of sequence length and shape

We compared JAX/flax against PyTorch and [Norse](https://github.com/norse/norse) to see how they stack up. Following the [benchmark code](https://github.com/norse/norse/tree/master/norse/benchmark) of Norse, we compared the speed of feeding data through a fully-connected linear network, varying sequence length, batch size and layer size. Using the tips provided [here](https://github.com/google/jax/discussions/8497), we find that JAX is faster for larger sequence lengths and matrices.

- `benchmark_jax_norse.py`: comparison between JAX/flax and Norse
- `benchmark_jax_torch_public.py`: comparison between JAX/flax and PyTorch

TODO:
- Also compare for C++ Norse ops and sparse tensors
- Look at/include new benchmark [code](https://github.com/norse/norse/tree/feature-benchmark/norse/benchmark) 

#### Hebbian learning with convolutions

Implementation of Hebbian learning with JAX, inspired by [Miconi 2021](http://arxiv.org/abs/2107.01729) and [Talloen et al.](http://arxiv.org/abs/2102.00428). If we have lifted custom JVPs in flax, we might be able to use a surrogate loss to imitate Hebbian learning, but for now we split convolutions into separate patches and keep traces manually. The learning itself doesn't really produce anything spectacular (maybe WTA is implemented incorrectly?), but the code works and is fast.

#### Other files:

- `comparison_jax.py`: contains an older comparison between JAX and PyTorch, which was the inspiration for this repo.
- `forward_ad.py`: attempt at comparing forward-mode autodiff for JAX and PyTorch (relevant issue [here](https://github.com/pytorch/pytorch/issues/10223))
- `benchmark_threshold.py`: benchmark for different kinds of spiking threshold implementations
- `toyexample_scan_var.py`: implementing scan with tracking state as variable
- `toyexample_scan.py`: using flax's lifted scan with state as input

## TODOs

The list here is not all; code also contains a lot of TODOs!

### JAX

- Test sparse tensor support
- Test forward-mode AD and RTRL/Snap-n
- Try out if `nn.scan` can improve loop performance when pushing spikes through the network