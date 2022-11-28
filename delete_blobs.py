import argparse
import json
from pathlib import Path

from azure_storage_utils import StorageAzureClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        choices=["after_container_detections", "after_pipeline"],
        help="the stage when to delete images",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="date when pipeline is run",
    )
    opt = parser.parse_args()

    saClient = StorageAzureClient(secret_key="data-storage-account-url")

    if opt.stage == "after_container_detections":
        if not Path(opt.date).exists():
            Path(opt.date).mkdir(exist_ok=True, parents=True)

        # download detections file from the storage account
        saClient.download_blob(
            cname="detections",
            blob_name=f"{opt.date}/empty_predictions.json",
            local_file_path=f"{opt.date}/empty_predictions.json",
        )

        input_file_path = f"{opt.date}/empty_predictions.json"
        f = open(input_file_path)
        input_data = json.load(f)

        images_to_remove = [opt.date + "/" + entry["pano_id"] for entry in input_data]

        print(
            f"Removed {len(images_to_remove)} images without a detection from the cloud."
        )

    if opt.stage == "after_pipeline":
        # For a container, delete all content from a certain date.
        cnames = ["blurred", "detections", "postprocessing-output"]

        for cname in cnames:
            all_data = saClient.list_container_content(
                cname=cname,
                blob_prefix=opt.date,
            )
            if all_data:
                saClient.delete_blobs(
                    cname=cname,
                    blob_names=all_data,
                )
                print(f"Removed {len(all_data)} blobs from container {cname}.")
            else:
                print(f"No blobs found in container {cname} for date {opt.date}.")
