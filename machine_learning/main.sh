#!/usr/bin/env bash

is_hpc=$1

condition="CRC"
declare -a profiles=( "OGs" "centrifuge" "KEGG_KOs" )
#declare -a MODELS=( "LOPO" "oLOPO_withCrossSupport" "oLOPO_withSignature" )


for profile in "${profiles[@]}"; do

    if [[ "$mode" == "train" ]]; then
        job_name="mlgut_${condition}_${profile}_${mode}"
        path="data/paper/${condition}_${profile}"
        mkdir -p ${path}
        err="${path}/${job_name}.err"
        out="${path}/${job_name}.out"
        if [[ "$is_hpc" == 1 ]]; then
            sbatch -J ${job_name} -N 1 -c 24 -e $err -o $out --wrap="ml anaconda2; conda activate ./.venv; python main.py ${condition} ${profile} ${path}"
        else
            conda activate ./.venv; python main.py ${condition} ${profile} ${path} >> ${out} 2>> ${err}
        fi
    fi

done
