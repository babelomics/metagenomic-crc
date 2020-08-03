#!/usr/bin/env bash

mode=$1

condition="CRC"
declare -a profiles=( "OGs" "centrifuge" "KEGG_KOs" )
job_name="mlgut${profile}"
path="data/paper/${condition}_${profile}"
mkdir -p ${path}
err="${path}/log.err"
out="${path}/log.out"

for profile in "${profiles[@]}"; do
    if [[ "$mode" == "train" ]]; then
        sbatch -J ${job_name} -N 1 -c 24 -e $err -o $out --wrap="ml anaconda2; conda activate ./.venv; python main.py ${condition} ${profile} ${path}"
    else
        sbatch -J ${job_name} -N 1 -c 24 -e $err -o $out --wrap="ml anaconda2; conda activate ./.venv; python significance.py ${condition} ${profile} ${path}"
    fi
done
