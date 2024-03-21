#!/bin/bash


DRIVE_LINK="https://drive.google.com/uc?id"

if command -v gdown &> /dev/null; then
    downloader="gdown"
else
    echo "gdown was not found. Please install gdown using pip."
fi


scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

mkdir -p "${scripts_dir}/../data" 

cd "${scripts_dir}/../data"  || exit

gdown "$DRIVE_LINK=1LHZGRgCcsEJVZVVkfZJLngqzJM8mtt1T" -O test_sample.csv
gdown "$DRIVE_LINK=1ya4xxP6TsTf1e1fCB_NzeLGp0zRUT1WV" -O test_sample_labels.csv



if [ $? -eq 0 ]; then
    echo "CSV files downloaded successfully."
else
    echo "Failed to download CSV files. Please check the link and try again."
fi


