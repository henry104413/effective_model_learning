# Effective model learning - bash launcher
# @author: Henry (henry104413)

# bash to carry out learning
# general run parameters set here
# advanced learning hyperparameters set in execute.py file
# code files musty be in the same directory 


# set below:
# 1) array of csv file names where pairs of columns are individual datasets,
# - any annotations will be skipped
# 2) experiment name to use in output filenames alongside the file name
# - !! now without the file extension (for easier output filename generation)
# 3) array of different numbers of defects to run with
# 4) array of repetitions numbers for each defect number
# 5) array of iterations (proposals) numbers for each defect numbers
# 6) proportion (as float) of data values from start to use for training; 1 means use all  
# note:
# 4) and 5) either have to be same length as 3),
# or length 1 if same settings to be used for each defect number
# !! but in each case must be arrays!  
declare -a target_csvs=("Wit-Fig4-6-0_025")
# "Wit-Fig4-5-0_1" "Wit-Fig4-6-0_025" "Wit-Fig4-6-0_1" "Wit-Fig4-6-0_2" "Wit-Fig4-7-0_1"
experiment_name="260422-4M"
defects_numbers=(2)
repetitions_numbers=(9)
iterations_numbers=(4000000) # ...match up with Ds
shock_anneal_ats=(0) # ...plural - match up with Ds
fix_tweak_width_ats=(1000000) # ...plural - match up with Ds
start_tweak_width_adaptation_ats=(100000) # ...plural - match up with Ds
fix_temperature_ats=(0) # ...plural - match up with Ds
start_temperature_adaptation_ats=(0) # ...plural - match up with Ds
proportion_training=1
configs=(11)
full=1
noise_stdevs=(0.01 0.05 0.1)
# note: iterations_number and shock_anneal_at and repetitions_number arrays allowed to be different for each number of defects,
# or if array of length one then this is always used; then repetitions carried out with identical setups
# note: same setups done for each specified noise_stdev; this is done for each of configs; this in turn is done for each target (usually only 1)
# so order is: each onfigs -> each noise_stdev -> each D -> corresponding or common R, iterations_number, shock_anneal_at 


# execution:
for target_csv in "${target_csvs[@]}"; do
for config in ${configs[@]}; do
	for noise_stdev in ${noise_stdevs[@]}; do
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
		    
		    # determine shock_anneal_at for this number of defects:
		    if [ ${#shock_anneal_ats[@]} -eq ${#defects_numbers[@]} ]; then
		    	shock_anneal_at=${shock_anneal_ats[i]}
		    elif [ ${#shock_anneal_ats[@]} -eq 1 ]; then
		    	shock_anneal_at=${shock_anneal_ats[0]}
		    else
		    	# if not specified properly, will pass 0
		    	# then execute file will not replace configs file value
		    	# note: means cannot anneal at iteration 0 (would be pointless anyway)
		    	shock_anneal_at=0	
		    fi
		    
		    # determine fix_tweak_width_at for this number of defects:
		    if [ ${#fix_tweak_width_ats[@]} -eq ${#defects_numbers[@]} ]; then
		    	fix_tweak_width_at=${fix_tweak_width_ats[i]}
		    elif [ ${#fix_tweak_width_ats[@]} -eq 1 ]; then
		    	fix_tweak_width_at=${fix_tweak_width_ats[0]}
		    else
		    	# if not specified properly, will pass 0
		    	# then execute file will not replace configs file value
		    	# note: means cannot tweak at iteration 0 (would be pointless anyway)
		    	fix_tweak_width_at=0	
		    fi
		    
		    # determine start_tweak_width_adaptation_at for this number of defects:
		    if [ ${#start_tweak_width_adaptation_ats[@]} -eq ${#defects_numbers[@]} ]; then
		    	start_tweak_width_adaptation_at=${start_tweak_width_adaptation_ats[i]}
		    elif [ ${#start_tweak_width_adaptation_ats[@]} -eq 1 ]; then
		    	start_tweak_width_adaptation_at=${start_tweak_width_adaptation_ats[0]}
		    else
		    	# if not specified properly, will pass 0
		    	# then execute file will not replace configs file value
		    	# note: means cannot tweak at iteration 0 (would be pointless anyway)
		    	start_tweak_width_adaptation_at=0	
		    fi
		    
		    # determine fix_temperature_at for this number of defects:
		    if [ ${#fix_temperature_ats[@]} -eq ${#defects_numbers[@]} ]; then
		    	fix_temperature_at=${fix_temperature_ats[i]}
		    elif [ ${#fix_temperature_ats[@]} -eq 1 ]; then
		    	fix_temperature_at=${fix_temperature_ats[0]}
		    else
		    	# if not specified properly, will pass 0
		    	# then execute file will not replace configs file value
		    	# note: means cannot tweak at iteration 0 (would be pointless anyway)
		    	fix_temperature_at=0	
		    fi
		    
		    # determine start_temperature_adaptation_at for this number of defects:
		    if [ ${#start_temperature_adaptation_ats[@]} -eq ${#defects_numbers[@]} ]; then
		    	start_temperature_adaptation_at=${start_temperature_adaptation_ats[i]}
		    elif [ ${#start_temperature_adaptation_ats[@]} -eq 1 ]; then
		    	start_temperature_adaptation_at=${start_temperature_adaptation_ats[0]}
		    else
		    	# if not specified properly, will pass 0
		    	# then execute file will not replace configs file value
		    	# note: means cannot tweak at iteration 0 (would be pointless anyway)
		    	start_temperature_adaptation_at=0	
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
			nohup python execute_learning_full.py "$target_csv" "$experiment_name" "$defects_number" "$rep" "$iterations_number" "$proportion_training" "$config" "$full" "$noise_stdev" "$shock_anneal_at" "$fix_tweak_width_at" "$start_tweak_width_adaptation_at" "$fix_temperature_at" "$start_temperature_adaptation_at" </dev/null &>"$experiment_name"_std"$noise_stdev"_"$target_csv"_conf"$config"_D"$defects_number"_R"$rep"_prog.txt & # regular
	#		python execute_learning_full.py "$target_csv" "$experiment_name" "$defects_number" "$rep" "$iterations_number" "$proportion_training" "$config" "$full" "$noise_stdev" "$shock_anneal_at" "$fix_tweak_width_at" "$start_tweak_width_adaptation_at" "$fix_temperature_at" "$start_temperature_adaptation_at" </dev/null &>"$experiment_name"_std"$noise_stdev"_"$target_csv"_conf"$config"_D"$defects_number"_R"$rep"_prog.txt # slurm
		    done
		done
	done
done
done
