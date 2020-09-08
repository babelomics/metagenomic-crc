#!/usr/bin/env bash

mode=$1

condition="CRC"
declare -a profiles=( "OGs" "centrifuge" "KEGG_KOs" )
declare -a MODELS=( "LOPO" "oLOPO_withCrossSupport" "oLOPO_withSignature" )

for profile in "${profiles[@]}"; do

    if [[ "$mode" == "train" ]]; then
        job_name="mlgut_${condition}_${profile}_${mode}"
        path="data/paper/${condition}_${profile}"
        mkdir -p ${path}
        err="${path}/${job_name}.err"
        out="${path}/${job_name}.out"
        sbatch -J ${job_name} -N 1 -c 24 -e $err -o $out --wrap="ml anaconda2; conda activate ./.venv; python main.py ${condition} ${profile} ${path}"
    fi
    
    if [[ "$mode" == "significance" ]]; then
        for model in "${MODELS[@]}"; do
            echo "$profile $model"
                job_name="mlgut_${condition}_${profile}_${mode}_${model}"
                path="data/paper/${condition}_${profile}"
                mkdir -p ${path}
                err="${path}/${job_name}.err"
                out="${path}/${job_name}.out"
            sbatch -J ${job_name} -N 1 -c 24 -e $err -o $out --wrap="ml anaconda2; conda activate ./.venv; python significance.py ${condition} ${profile} train $model ${path}"
        done
    fi
    
    if [[ "$mode" == "adenoma" ]]; then
        job_name="mlgut_${condition}_${profile}_${mode}"
        path="data/paper/${condition}_${profile}"
        mkdir -p ${path}
        err="${path}/${job_name}.err"
        out="${path}/${job_name}.out"
        sbatch -J ${job_name} -N 1 -c 24 -e $err -o $out --wrap="ml anaconda2; conda activate ./.venv; python run_adenoma.py ${condition} ${profile} ${path}"
    fi
    
done
