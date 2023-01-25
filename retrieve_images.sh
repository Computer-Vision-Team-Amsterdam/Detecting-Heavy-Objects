#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No argument provided. Please provide the path to the source directory as an argument. For example 2022-12-31 21:00:00.00"
    exit 1
fi

cloudvps_folder=$(echo $1 | cut -d ' ' -f1)
cloudvps_folder=$(echo $cloudvps_folder | sed 's/-/\//g')
echo "the Azure destination folder is: $cloudvps_folder"

azure_folder=$(echo $1 | cut -d '.' -f1)
azure_folder=$(echo $azure_folder | sed 's/ /_/g' | sed 's/-/:/g' | sed 's/:/-/g')
echo "the Azure destination folder is: $azure_folder"

# Log in to Azure using a managed identity
az login --identity --username $USER_ASSIGNED_MANAGED_IDENTITY

# Get the key vault URL from the environment variable
keyVaultUrl=$(echo $AIRFLOW__SECRETS__BACKEND_KWARGS | jq -r '.vault_url')

# Extract the key vault name from the URL
keyVaultName=$(echo $keyVaultUrl | grep -oP '(?<=https://)[^.]+(?=.vault)')

# Set the storage account name and container name
storageAccountUrl=$(az keyvault secret show --vault-name $keyVaultName -n "data-storage-account-url" --query "value" -o tsv)
storageAccountName=$(echo $storageAccountUrl | grep -oP '(?<=https://)[^.]+(?=.blob)')

secretTenant=$(az keyvault secret show --vault-name $keyVaultName -n "CloudVpsBlurredTenant" --query "value" -o tsv)
secretUser=$(az keyvault secret show --vault-name $keyVaultName -n "CloudVpsBlurredUsernameShort" --query "value" -o tsv)
secretKey=$(az keyvault secret show --vault-name $keyVaultName -n "CloudVpsBlurredPassword" --query "value" -o tsv)

# Create the rclone configurations
rclone config create objectstore_rclone swift auth https://identity.stack.cloudvps.com/v2.0 auth_version 2 tenant $secretTenant user $secretUser key $secretKey --quiet > /dev/null
containerName="unblurred"
rclone config create azureblob_rclone azureblob container=$containerName account=$storageAccountName --quiet > /dev/null
containerNameTwee="retrieve-images-input"
rclone config create azureblob_rclone_twee azureblob container=$containerNameTwee account=$storageAccountName --quiet > /dev/null

# Set the source and destination directories
src_dir=objectstore_rclone:panorama/$cloudvps_folder
dst_dir=azureblob_rclone:unblurred/$azure_folder/


# Get directory structure for source dir, and remove the first line ("/") and stripansi
rclone tree $src_dir --noindent --include "equirectangular/panorama_8000.jpg" --noreport | sed -e '1,1d' -e 's/\x1B\[[0-9;]*[JKmsu]//g' > files.txt

# Create paths from tree output
# e.g. TMX7316010203-002927/pano_0015_000036/equirectangular/panorama_8000.jpg
awk '{
   if ($0 ~ /^TMX/) {
     prefix = $0;
   } else if ($0 ~ /^pano/) {
     path = $0;
   } else if ($0 ~ /^equirectangular/) {
     print prefix "/" path "/" $0 "/panorama_8000.jpg";
   }
}' files.txt > paths.txt

# Convert Cloud VPS paths to pano ids
sed 's/\/equirectangular\/panorama_8000.jpg//' paths.txt | tr '/' '_' > pano_ids.txt

counter=0
## Loop over the combined paths and panorama id files to get the source and destination file path pairs
#paste paths.txt pano_ids.txt | while read path pano_id; do
#   rclone copyto $src_dir/$path $dst_dir$pano_id.jpg \
#      --azureblob-use-msi \
#      --azureblob-msi-client-id=$USER_ASSIGNED_MANAGED_IDENTITY;
#
#      counter=$((counter+1))
#      if [ $counter -eq 10 ]; then
#          break
#      fi
#done

while read line; do
    # Convert Cloud VPS paths to pano ids
    newline=$(echo "$line" | sed 's/\/equirectangular\/panorama_8000.jpg//' | tr '/' '_')
    echo $newline
    rclone copyto "$src_dir/$line" "$dst_dir$newline.jpg" \
    --azureblob-use-msi \
    --azureblob-msi-client-id=$USER_ASSIGNED_MANAGED_IDENTITY;

    # check the exit status of the rclone copyto command after every iteration
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy $line to $dst_dir$newline.jpg"
    fi

    counter=$((counter+1))
    if [ $counter -eq 10 ]; then
        break
    fi
done < paths.txt

num_workers=8 # TODO
chunk_folder="splits"
mkdir $chunk_folder

dst_dir2=azureblob_rclone_twee:retrieve-images-input/$azure_folder

split -n $num_workers pano_ids.txt $chunk_folder/

i=1
for file in $chunk_folder/*; do
    mv "$file" "$chunk_folder/$i.txt"
    i=$((i+1))
done

for chunk_file in "$chunk_folder/*"
do
    base_name=$chunk_file
    echo $base_name
    rclone copyto $chunk_file $dst_dir2/$base_name \
        --azureblob-use-msi \
        --azureblob-msi-client-id=$USER_ASSIGNED_MANAGED_IDENTITY
    # check the exit status of the rclone copyto command after every iteration
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy $chunk_file to $dst_dir2/$base_name"
    fi
done