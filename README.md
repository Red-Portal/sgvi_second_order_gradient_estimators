
# SGVI with Second-Order Gradient Estimators

This repo contains the code for replicating the experimental results in the paper:

```
Stochastic Gradient Variational Inference with Second-Order Gradient Estimators from Bures-Wasserstein to Parameter Space
```

## Installation

The dependencies can be installed by executing the following code in the root of the project directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Code

The main script files are the following:

* `run_experiments.jl`: All of the experimental results can be run by executing the script 
* `scratch.jl`: A simple example on how to run the VI algorithms.
* `process_data.jl`: The code for processing the raw data contained in the HDF files.

