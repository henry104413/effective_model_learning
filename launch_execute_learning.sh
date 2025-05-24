# Effective model learning - bash launcher
# @author: Henry (henry104413)

# bash to carry out learning
# general run parameters set here
# advanced learning hyperparameters set in execute.py file
# code files musty be in the same directory 


# set below:
# 1) target csv file where pairs of columns are individual datasets,
# - any annotations will be skipped
# 2) experiment name to use in output filenames
# 3) array of different numbers of defects to run with
# 4) array of repetitions numbers for each defect number
# 5) array of iterations (proposals) numbers for each defect numbers
# 6) proportion (as float) of data values from start to use for training; 1 means use all  
# note:
# 4) and 5) either have to be same length as 3),
# or length 1 if same settings to be used for each defect number
# but in each case must be arrays!  
target_csv="Witnessing_Fig4b.csv"
experiment_name="250524-fig-run3"
defects_numbers=(0 1)
repetitions_numbers=(4)
iterations_numbers=(30)
proportion_training=0.7


# execution:
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
	nohup python execute_learning.py "$target_csv" "$experiment_name" "$defects_number" "$rep" "$iterations_number" "$proportion_training"  </dev/null &>"$experiment_name"_D"$defects_number"_R"$rep"_prog.txt &
	done
done

