# Effective model learning - bash launcher
# @author: Henry (henry104413)

# bash to sort model files into separate folders given clusters count
# according to existing cluster assignment dictionary 

# model files must be in this folder and follow naming convention:
# <experiment name>_D<number of defects>_R<run number>_<which output>.<type>
# alongside the assignment lists produced by sort_by_clusters.py
# <experiment name>_D<number of defects>_Cs<clusters count>_C<cluster number>.txt

# for now to be called separately for each defects number
# later will allow array specification



experiment_name="quick-test"
defects_number=1
clusters_count=3


python sort_by_clusters.py "$experiment_name" "$defects_number" "$clusters_count" 

assignment_name="$experiment_name"_D"$defects_number"_Cs"$clusters_count"

mkdir "$assignment_name"

for ((cluster=0; cluster<clusters_count; cluster++)); do

    # make directory for each cluster; in it goes champion and subdirectory with all cluster members
    list="$assignment_name"_C"$cluster".txt
    mkdir "$assignment_name"/"$assignment_name"_C"$cluster"
    mkdir "$assignment_name"/"$assignment_name"_C"$cluster"/"$assignment_name"_C"$cluster"_all
    
    # copy model files from current to corresponding directories:
    i=1
    while IFS= read -r line; do
        if [ $i -eq 1 ]; then
            cp $line* "$assignment_name"/"$assignment_name"_C"$cluster"
            echo first: copying "$line"
        else
            cp $line* "$assignment_name"/"$assignment_name"_C"$cluster"/"$assignment_name"_C"$cluster"_all
            echo others: copying "$line"
        fi 
        ((++i))
    done < "$assignment_name"_C"$cluster".txt
done    
    
    
    
