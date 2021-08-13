#!/usr/bin/env bash


file_folder=$(dirname "$0")
project_folder="${file_folder}/.."
dotenv_path="${project_folder}/.env"
# Read .env file
export $(egrep -v '^#' ${dotenv_path} | xargs)

raw_data_folder="${file_folder}/../data/raw/"
mkdir -p $raw_data_folder

source="$user_id@$ssh_data_path/ngless_samples_profiles"
target=${raw_data_folder}

#TODO use secretes from .env
rsync -r -a -v -e ssh $source $target

source="$user_id@$ssh_data_path/centrifuge_samples_profiles"
target=${raw_data_folder}

#TODO use secretes from .env
rsync -r -a -v -e ssh $source $target

source="$user_id@$metadata_path"
target=${raw_data_folder}

#TODO use secretes from .env
rsync -r -a -v -e ssh --prune-empty-dirs --include "*/" --include="*subset.tsv" --exclude="*" $source $target
mv "${target}/machine_learning" "${target}/metadata"
find "${target}/metadata" -type d -empty -delete

scp $user_id@$kegg_dict_file $raw_data_folder
