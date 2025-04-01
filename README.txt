Author: henry104413

Under continuous development.

Implements quantum system models consisting of two-level subsystems (TLSs) representing qubits and defects. Allows single-site Lindblad processes and coherent couplings between any pairs of TLSs. Includes methods to generate dynamics with respect to specified observables (currently from a predefined set).

Performs model learning given target data (times, observables, and measurements with respect to those). Uses a reversible-jump Markov chain Monte Carlo algorithm. Based on user-specified hyperparameters, random moves are carried out and accepted or rejected via a Metropolis-Hastings-style criterion. These include adding or removing Lindblad processes or couplings from a predefined library as well as tweaking existing parameters.

Files starting with "execute" are examples of scripts that can be run.

Developed for python 3.11.5, virtual environment and container with dependencies shall be provided in the future.
