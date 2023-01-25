#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Error: No two arguments provided. Please provide the path to the source directory as an argument. For example 2022-12-31 21:00:00.00"
    exit 1
fi

merged="$1 $2"
azure_folder=$(date -d "$merged" +"%Y-%m-%d_%H-%M-%S")
echo azure_folder

cloudvps_folder=$(echo $1 | sed 's/-/\//g')
echo cloudvps_folder

# Log in to Azure using a managed identity
az login --identity --username $USER_ASSIGNED_MANAGED_IDENTITY

# Get the key vault URL from the environment variable
keyVaultUrl=$(echo $AIRFLOW__SECRETS__BACKEND_KWARGS | jq -r '.vault_url')

# Extract the key vault name from the URL
keyVaultName=$(echo $keyVaultUrl | grep -oP '(?<=https://)[^.]+(?=.vault)')

# Set the storage account name and container name
storageAccountUrl=$(az keyvault secret show --vault-name $keyVaultName -n "data-storage-account-url" --query "value" -o tsv)
containerName="unblurred"
storageAccountName=$(echo $storageAccountUrl | grep -oP '(?<=https://)[^.]+(?=.blob)')

secretTenant=$(az keyvault secret show --vault-name $keyVaultName -n "CloudVpsBlurredTenant" --query "value" -o tsv)
secretUser=$(az keyvault secret show --vault-name $keyVaultName -n "CloudVpsBlurredUsernameShort" --query "value" -o tsv)
secretKey=$(az keyvault secret show --vault-name $keyVaultName -n "CloudVpsBlurredPassword" --query "value" -o tsv)

# Create the rclone configurations
rclone config create objectstore_rclone swift auth https://identity.stack.cloudvps.com/v2.0 auth_version 2 tenant $secretTenant user $secretUser key $secretKey --quiet > /dev/null
rclone config create azureblob_rclone azureblob container=$containerName account=$storageAccountName --quiet > /dev/null

# Set the source and destination directories
src_dir=objectstore_rclone:panorama/$cloudvps_folder
dst_dir=my_folder/

# Set the maximum number of retries
MAX_RETRIES=2
counter=0

# Loop through all top-level directories in the source directory
for dir1 in $(rclone lsf --dirs-only $src_dir); do
    dir_run_name=$(basename $dir1)

    # Loop through all subdirectories in the source directory
    for dir2 in $(rclone lsf --dirs-only $src_dir/$dir1); do
        # Get the parent directory name
        parent_dir_name=$(basename $dir2)
        input_folder=$src_dir/$dir1$dir2
        # Loop through all files in the equirectangular directory

        # Loop through all files in the equirectangular directory that are currently present in the server at that moment.
        files=$(rclone lsl $input_folder --include "equirectangular/panorama_8000.jpg")


        if [[ -z "$files" ]]; then
            echo "No files found"
        else
            while read -r file_date file_time file_name; do
                # echo "Date: $file_date"
                # echo "Time: $file_time"
                # echo "File: $file_name"


                # Try to copy the file using the parent directory name
                retries=0
                out_file_name=$dir_run_name"_"$parent_dir_name.jpg

                counter=$((counter+1))
                if [ $counter -eq 10 ]; then
                    break
                fi

                # Maybe with --no-traverse and --transfers 
                while ! rclone copyto $input_folder$file_name $dst_dir$out_file_name; do
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
            done < <(echo "$files" | awk '{print $2, $3, $4}')
        fi
        break
    done

done

src_dir2=my_folder/
dst_dir2=azureblob_rclone:unblurred/$azure_folder

# List all files in the source folder
files=$(rclone ls $src_dir2)

if [[ -z "$files" ]]; then
    echo "No files found"
else
    while read -r file_size file_name; do
        # echo "Date: $file_date"
        # echo "Time: $file_time"
        # echo "File: $file_name"
        echo $file_name

        # Try to copy the file using the parent directory name
        retries=0

        # Maybe with --no-traverse and --transfers 
        rclone copy $src_dir2$file_name $dst_dir2 --azureblob-use-msi --azureblob-msi-client-id=$USER_ASSIGNED_MANAGED_IDENTITY
        echo "hallo"

    done < <(echo "$files" | awk '{print $1, $2}')
fi