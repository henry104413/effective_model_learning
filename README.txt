Author: henry104413

Under continuous development.


Implements quantum system models consisting of two-level subsystems (TLSs) representing qubits and defects. Allows single-site Lindblad processes and coherent couplings between any pairs of TLSs. Includes methods to generate dynamics according to the Lindblad master equation, with respect to specified observables currently from a predefined set.

Performs model learning given target data (times, observables, and measurements with respect to those). Uses a reversible-jump Markov chain Monte Carlo algorithm. Based on user-specified hyperparameters, random moves are carried out and accepted or rejected via a Metropolis-Hastings-style criterion. These include adding or removing Lindblad processes or couplings from a predefined library as well as tweaking existing model parameters.


The file "execute_learning.py" can be run and takes command line arguments for run settings. The bash script "launch_execute_learning.sh" allows easily running batches with different settings. Advanced learning hyperparameters can be tuned inside "execute_learning.py". The "find_clusters.py" file is also runnable with command line arguments and currently performs and evaluates k-means clustering on a set of imported learned models across a set of k values - can be run using the bash script "launch_find_clusters.sh" for easy parameter setup. Once completed, "sort_by_clusters.py" can sort the output files of the learned models into separate directories according to the best cluster assignment identified for a given clusters count (k) - with the bash script "launch_sort_by_clusters.sh" again enabling convenient setup.


Developed for python 3.11.5. A working conda environment "effective_model_learning_conda_env.yml" is provided and a container shall follow in the future.

Dependencies include: numpy, scipy, qutip, matplotlib; additionally for clustering: sklearn and knee.

Works well with the following combination of packeges.
Caution: Many of these have poor backwards compatibility.
Using a different - even newer - version of any one may cause errors.
python=3.11.5
numpy=1.24.3
scipy=1.12.0
qutip=4.7.5
# ensure qutip doesn't overwrite scipy upon installation
matplotlib=3.7.2
scikit-learn=1.6.1
kneed=0.8.5


