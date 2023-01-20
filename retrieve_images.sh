#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No argument provided. Please provide the path to the source directory as an argument. For example 2022/12/31"
    exit 1
fi

# Set the source and destination directories
src_dir=objectstore_rclone:panorama/$1
dst_dir=my_folder/

# Set the maximum number of retries
MAX_RETRIES=2
# Define the config option
RCLONE_CONF="--config rclone.conf"

# Loop through all top-level directories in the source directory
for dir1 in $(rclone lsf --dirs-only $src_dir $RCLONE_CONF); do
    dir_run_name=$(basename $dir1)

    # Loop through all subdirectories in the source directory
    for dir2 in $(rclone lsf --dirs-only $src_dir/$dir1 $RCLONE_CONF); do
        # Get the parent directory name
        parent_dir_name=$(basename $dir2)
        input_folder=$src_dir/$dir1$dir2
        # Loop through all files in the equirectangular directory
        for file in $(rclone ls $input_folder --include "equirectangular/panorama_8000.jpg" $RCLONE_CONF); do
            # Try to copy the file using the parent directory name
            retries=0
            out_file_name=$dir_run_name"_"$parent_dir_name.jpg
            # Maybe with --no-traverse 
            while ! rclone copyto $input_folder$file $dst_dir$out_file_name $RCLONE_CONF; do
                # Handle connection errors
                if [ $? -eq 1 ]; then
                    echo "Error: Connection failed. Retrying in 10 seconds..."
                else
                    echo "Error: Failed to download file ${out_file_name}. Retrying in 10 seconds..."
                fi
                # Check if the maximum number of retries has been reached
                if [ "$retries" -ge "$MAX_RETRIES" ]; then
                    echo "Error: Maximum number of retries reached. Exiting..."
                    break 2
                fi
                retries=$((retries+1))
                sleep 10
            done
        done
    done
done