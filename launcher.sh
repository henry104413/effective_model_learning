# settings:
experiment_name="justatest2_"
defects_numbers=(1 2 3)
repetitions=1
iterations_numbers=(500 600 700)
# note:
# if maximum iterations different for each defects number:
# specify as array of same length as defects_numbers;
# if same for all of them: 
# use array of length 1 - but must be an array!

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
	nohup python execute.py "$experiment_name" "$defects_number" "$rep" "$iterations_number"  </dev/null &>"$experiment_name"_D"$defects_number"_R"$rep"_prog.txt &
	done
done

