# Effective model learning - bash launcher
# @author: Henry (henry104413)

# bash to find clusters for given clusters counts

# pickled models must be in this folder
# and must follow naming convention:
# <experiment name>_D<number of defects>_R<repetition>_best.pickle

# repetitions are iterated through from 1 up to repetitions number

# minimum clusters should be at least 2
# maximum should be at most number of points (models) - 1
# (both required for silhouette score calculation)
# note: it follows that at least 3 points required

# settings:
# last three (= 5), 6), 7) if joined with execute launcher) 
# each either array of same length as defects numbers
# or array of length one (then taken same for all defects numbers)
experiment_name="quick-test"
defects_numbers=(1)
repetitions_numbers=(10)
mins_clusters=(2)
maxs_clusters=(8)


for i in ${!defects_numbers[@]}; do
    defects_number=${defects_numbers[i]}

    # determine repetitions number for this number of defects:
    if [ ${#repetitions_numbers[@]} -eq ${#defects_numbers[@]} ]; then
    	repetitions_number=${repetitions_numbers[i]}
    elif [ ${#repetitions_numbers[@]} -eq 1 ]; then
    	repetitions_number=${repetitions_numbers[0]}
    else
    	# if not specified properly, set default here:
    	repetitions_number=3	
    fi
    
    # determine minimum clusters for this number of defects:
    if [ ${#mins_clusters[@]} -eq ${#defects_numbers[@]} ]; then
    	min_clusters=${mins_clusters[i]}
    elif [ ${#mins_clusters[@]} -eq 1 ]; then
    	min_clusters=${mins_clusters[0]}
    else
    	# if not specified properly, set default here:
    	min_clusters=2	
    fi
    
    # determine maximum clusters number for this number of defects:
    if [ ${#maxs_clusters[@]} -eq ${#defects_numbers[@]} ]; then
    	max_clusters=${maxs_clusters[i]}
    elif [ ${#repetitions_numbers[@]} -eq 1 ]; then
    	max_clusters=${maxs_clusters[0]}
    else
    	# if not specified properly, set default here:
    	max_clusters=$(($repetitions_number - 1))
    fi

    # execution:
    nohup python find_clusters.py "$experiment_name" "$defects_number" "$repetitions_number" "$min_clusters" "$max_clusters"  </dev/null &>"$experiment_name"_D"$defects_number"_clustering_prog.txt &
    
done

