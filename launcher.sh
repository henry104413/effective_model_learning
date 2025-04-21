# Effective model learning - bash launcher
# @author: Henry (henry104413)

# bash to carry out learning
# general run parameters set here
# advanced learning hyperparameters set in execute.py file
# code files musty be in the same directory 

# set below:
# 1) experiment name to use in output filenames
# 2) target csv file where pairs of columns are individual datasets,
# - any annotations will be skipped
# 3) array of different numbers of defects to run with
# 4) array of repetitions numbers for each defect number
# 5) array of iterations (proposals) numbers for each defect numbers
# note:
# 4) and 5) either have to be same length as 3),
# or length 1 if same settings to be used for each defect number
# but in each case must be arrays!  
experiment_name="justatest2_"
target_csv=""
defects_numbers=(1 2 3)
repetitions=1
iterations_numbers=(500 600 700 800)

# execution:
for i in ${!defects_numbers[@]}; do
    defects_number=${defects_numbers[i]}
    if [ ${#iterations_numbers[@]} -eq ${#defects_numbers[@]} ]; then
    	iterations_number=${iterations_numbers[i]}
    elif [ ${#iterations_numbers[@]} -eq 1 ]; then
    	iterations_number=${iterations_numbers[0]}
    else
    	# if not specified properly, will pass 0
    	# then execute file will use its default value
    	iterations_number=0	
    fi
    for ((rep=1; rep<=repetitions; rep++)); do
	echo launching for $defects_number defects repetition no. $rep
	echo iterations number $iterations_number
	nohup python execute.py "$experiment_name" "$target_csv" "$defects_number" "$rep" "$iterations_number"  </dev/null &>"$experiment_name"_D"$defects_number"_R"$rep"_prog.txt &
	done
done

