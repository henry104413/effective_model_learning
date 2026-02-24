Author: henry104413

Under continuous development.


Implements open quantum system models consisting of two-level subsystems (TLSs) representing system of interest (here referred to as qubit) and virtual systems (here called defects). Allows single-site Lindblad processes and coherent couplings between any pairs of TLSs. Includes methods to generate dynamics according to the Lindblad master equation, with respect to specified observables measured on the qubit. Operators for processes including couplings, as well as observables, are currently taken from a user-defined library.

Performs model learning given target data (times alongside observables and measurements with respect to those). Uses a reversible-jump Markov chain Monte Carlo algorithm. Based on user-specified hyperparameters, random proposals are carried out and accepted or rejected via a Metropolis-Hastings criterion. These include the addition or removal of a Lindblad process or coupling as permitted by the library (collectively known as reversible jumps), and tweaking simulatenously of all current model parameters.


The python file "execute_learning_full.py" executes a single chain and takes command line arguments for run settings. The bash script "launch.sh" allows easily running batches of chains with different main settings. Advanced learning hyperparameters can be tuned inside "configs.py". The python file "collate_cluster.py" file is also executable: it collates accepted models form a set of chains, outputs their vectorised version and executes k-means clustering over a range of k values. Once completed, k has to be manually set in the python executable "analyse.py", which computes correlations and hierarchichal clustering of parameters, plots cluster popularity, champion vectors, and calculates mean predictions for each cluster and overall.


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


