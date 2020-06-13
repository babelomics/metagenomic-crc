condition="CRC"
profile="OGs"
job_name="mlgut${profile_name}"
path="data/paper/${condition}_${profile_name}"
mkdir -p ${path}
err="${path}/log.err"
out="${path}/log.out"

sbatch -J ${job_name} -N 1 -c 24 -e $err -o $out --wrap="ml anaconda2; conda activate ./.venv; python main.py ${condition} ${profile} ${path}"
