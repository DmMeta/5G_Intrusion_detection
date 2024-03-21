#!/bin/bash


if command -v gdown &> /dev/null; then
    downloader="gdown"
else
    echo "gdown was not found. Please install gdown using pip."
fi

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

#assuming it is run within the scripts directory
gdown --folder "1rB0CKVzsqV-nYz90au2a9n4Nh66RAja9" -O "${scripts_dir}/../"



if [ $? -eq 0 ]; then
    echo "CSV files downloaded successfully."
else
    echo "Failed to download CSV files. Please check the link and try again."
fi



