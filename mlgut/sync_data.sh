#!/usr/bin/env bash

user_id=$1
data_path=$2
metadata_path=$3
kegg_dict_file=$4
is_ssh=$5

file_folder=$(dirname "$0")
raw_data_folder="${file_folder}/../data/raw/"
mkdir -p $raw_data_folder

source="$user_id@$data_path/ngless_samples_profiles"
target=${raw_data_folder}

#TODO use secretes from .env
rsync -r -a -v -e ssh $source $target

source="cloucera@$data_path/centrifuge_samples_profiles"
target=${raw_data_folder}

#TODO use secretes from .env
rsync -r -a -v -e ssh $source $target

source="cloucera@$metadata_path"
target=${raw_data_folder}

#TODO use secretes from .env
rsync -r -a -v -e ssh --prune-empty-dirs --include "*/" --include="*subset.tsv" --exclude="*" $source $target
mv "${target}/machine_learning" "${target}/metadata"
find "${target}/metadata" -type d -empty -delete

scp cloucera@$kegg_dict_file $raw_data_folder
