import argparse
from pathlib import Path
import json

from azure_storage_utils import StorageAzureClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        choices=["no_detections", "all_data"],
        help="data to delete",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="date when pipeline is run",
    )
    opt = parser.parse_args()

    saClient = StorageAzureClient(secret_key="data-storage-account-url")

    if opt.stage == "no_detections":
        if not Path(opt.date).exists():
            Path(opt.date).mkdir(exist_ok=True, parents=True)

        # download detections file from the storage account
        input_file_path = f"{opt.date}/coco_instances_results.json"
        saClient.download_blob(
            cname="detections",
            blob_name=f"{opt.date}/coco_instances_results.json",
            local_file_path=f"{opt.date}/coco_instances_results.json",
        )

        f = open(input_file_path)
        input_data = json.load(f)

        # Get all images with detections (with duplicates)
        images_w_det = [item["pano_id"] for item in input_data]

        images_all = saClient.list_container_content(
            cname="blurred",
            blob_prefix=opt.date,
        )

        images_to_remove = set(images_all) - set(images_w_det)
        saClient.delete_blob(
            cname="blurred",
            blob_names=images_to_remove,
            blob_prefix=opt.date,
        )
        print(f"Removed {len(images_to_remove)} images without a detection from the cloud.")

    if opt.stage == "all_data":
        # For a container, delete all content from a certain date.
        cnames = ["blurred", "detections", "postprocessing-output"]

        for cname in cnames:
            # TODO what if empty?
            all_data = saClient.list_container_content(
                cname=cname,
                blob_prefix=opt.date,
            )

            saClient.delete_blob(
                cname=cname,
                blob_names=all_data,
            )

            print(f"Removed {len(all_data)} blobs from container {cname}.")
