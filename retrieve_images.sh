#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No argument provided. Please provide the path to the source directory as an argument. For example 2022/12/31/"
    exit 1
fi

# Log in to Azure using a managed identity
az login --identity --username $USER_ASSIGNED_MANAGED_IDENTITY --verbose --debug

# Get the key vault URL from the environment variable
keyVaultUrl=$(echo $AIRFLOW__SECRETS__BACKEND_KWARGS | jq -r 'vault_url')

# Extract the key vault name from the URL
keyVaultName=$(echo $keyVaultUrl | awk -F/ '{print $4}')

echo keyVaultName

# Set the secret name
secretName="chris_test"

# Retrieve the secret value
secretValue=$(az keyvault secret show --vault-name $keyVaultName -n $secretName --query "value" -o tsv)

# Use the secret value in your script
echo "The secret value is: $secretValue"

exit 1

# Set the storage account name and container name
storageAccountName=$secretValue
containerName="unblurred"

secretTenant="test"
secretUser="test"
secretKey="test"

# Create the rclone configurations
rclone config create azureblob_rclone azureblob \
  --azureblob-account=$storageAccountName \
  --azureblob-container=$containerName \
  --azureblob-use-msi
rclone config create objectstore_rclone swift auth https://identity.stack.cloudvps.com/v2.0 auth_version 2 tenant $secretTenant user $secretUser key $secretKey


# Set the source and destination directories
src_dir=objectstore_rclone:panorama/$1
dst_dir=azureblob_rclone:my_folder/

# Set the maximum number of retries
MAX_RETRIES=2

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

        echo $files

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
    done
done