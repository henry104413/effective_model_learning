# Effective model learning
# @author: Henry (henry104413)

# bash to execute learning and find clusters
# general run parameters set here
# advanced learning hyperparameters set in execute_learning.py file
# code files and data must be in the same directory 

trap '' HUP

# set below:
# 1) target csv file where pairs of columns are individual datasets,
# - any annotations will be skipped, now using first dataset by default
# 2) experiment name to use in output filenames
# 3) array of different numbers of defects to run with
# 4) array of repetitions numbers for each defects number
# 5) array of iterations (proposals) numbers for each defects number
# 6) 
# note:
# 4), 5), 6), 7) either have to be same length as 3),
# or length 1 if same settings to be used for each defects number
# i. e., each either array of same length as defects numbers
# or array of length one (then taken same across all defects numbers)
# - in any case must be arrays!  
target_csv="Witnessing_Fig4b.csv"
experiment_name="quick-test"
defects_numbers=(1)
repetitions_numbers=(2)
iterations_numbers=(2000)
mins_clusters=(2)
maxs_clusters=(15)

# important notes:
# naming convention for best model loadable files:
# <experiment name>_D<number of defects>_R<repetition>_best.pickle
# repetitions are iterated through from 1 up to repetitions number
# minimum clusters should be at least 2
# maximum should be at most number of points (models) - 1
# (both required for silhouette score calculation)
# nb: it follows that at least 3 points required

# trackers to check whether all learning runs completed:
touch started_tracker ; rm started_tracker
touch finished_tracker ; rm finished_tracker


# launch parallel learning runs:
for i in ${!defects_numbers[@]}; do
    defects_number=${defects_numbers[i]}

    # determine iterations number for this number of defects:
    if [ ${#iterations_numbers[@]} -eq ${#defects_numbers[@]} ]; then
    	iterations_number=${iterations_numbers[i]}
    elif [ ${#iterations_numbers[@]} -eq 1 ]; then
    	iterations_number=${iterations_numbers[0]}
    else
    	# if not specified properly, will pass 0
    	# then execute file will use its default value
    	iterations_number=0	
    fi
    
    # determine repetitions number for this number of defects:
    if [ ${#repetitions_numbers[@]} -eq ${#defects_numbers[@]} ]; then
    	repetitions_number=${repetitions_numbers[i]}
    elif [ ${#repetitions_numbers[@]} -eq 1 ]; then
    	repetitions_number=${repetitions_numbers[0]}
    else
    	# if not specified properly, set default here:
    	repetitions_number=3
    fi
    
    for ((rep=1; rep<=repetitions_number; rep++)); do
	echo launching for $defects_number defects repetition no. $rep
	echo "started" >> started_tracker
	(python execute_learning.py "$target_csv" "$experiment_name" "$defects_number" "$rep" "$iterations_number"  </dev/null &>"$experiment_name"_D"$defects_number"_R"$rep"_prog.txt ; echo "finished" >> finished_tracker) &
	done
done


# keep checking trackers until all learning runs have finished:
while [ $(cat started_tracker|wc -l) -gt $(cat finished_tracker|wc -l) ] ; do
    sleep 1 # in seconds
done


# clustering:
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
    python find_clusters.py "$experiment_name" "$defects_number" "$repetitions_number" "$min_clusters" "$max_clusters"  </dev/null &>"$experiment_name"_D"$defects_number"_clustering_prog.txt &
    
done


