# Effective model learning - bash launcher
# @author: Henry (henry104413)

# bash to find clusters for given clusters counts

# pickled models must be in this folder
# and must follow naming convention:
# <experiment name>_D<number of defects>_R<run number>_best.pickle

# run numbers are iterated through from 1 up to runs number bound

# minimum clusters should be at least 2
# maximum should be at most number of points (models) - 1
# (both required for silhouette score calculation)

# for now to be called separately for each defects number
# later will allow array specification


# settings:
experiment_name="test_Wit_Fig4b-grey"
defects_number=1
run_number_bound=30
min_clusters=2
max_clusters=10


# execution:
nohup python find_clusters.py "$experiment_name" "$defects_number" "$run_number_bound" "$min_clusters" "$max_clusters"  </dev/null &>"$experiment_name"_D"$defects_number"_clustering_prog.txt &

