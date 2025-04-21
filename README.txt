Author: henry104413

Under continuous development.

Implements quantum system models consisting of two-level subsystems (TLSs) representing qubits and defects. Allows single-site Lindblad processes and coherent couplings between any pairs of TLSs. Includes methods to generate dynamics with respect to specified observables (currently from a predefined set).

Performs model learning given target data (times, observables, and measurements with respect to those). Uses a reversible-jump Markov chain Monte Carlo algorithm. Based on user-specified hyperparameters, random moves are carried out and accepted or rejected via a Metropolis-Hastings-style criterion. These include adding or removing Lindblad processes or couplings from a predefined library as well as tweaking existing parameters.

The file "execute.py" can be run and takes command line arguments for run settings. The bash script "launcher.sh" allows easily running batches with different settings. Advanced learning hyperparameters can be tuned in the execute file. The "clustering.py" file is runnable and currently performs and evaluates k-means clustering on a set of imported learned models.

Developed for python 3.11.5, virtual environment and container with dependencies shall be provided in the future.

Dependencies include: numpy, scipy, qutip, matplotlib; also sklearn and knee for the clustering. 
