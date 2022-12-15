import argparse
import json

from utils.azure_storage import StorageAzureClient
from utils.date import get_start_date

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        choices=["after_container_detections", "after_pipeline"],
        help="The stage when to delete images",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Processing date in the format %Y-%m-%d %H:%M:%S.%f",
    )
    opt = parser.parse_args()

    saClient = StorageAzureClient(secret_key="data-storage-account-url")

    start_date_dag, start_date_dag_ymd = get_start_date(opt.date)

    if opt.stage == "after_container_detections":
        input_file_path = "empty_predictions.json"

        # download detections file from the storage account
        saClient.download_blob(
            cname="detections",
            blob_name=f"{start_date_dag}/empty_predictions.json",
            local_file_path=input_file_path,
        )

        f = open(input_file_path)
        input_data = json.load(f)

        images_to_remove = [
            start_date_dag + "/" + entry["pano_id"] for entry in input_data
        ]

        # TODO remove, validate that images are also in blob container

        print(
            f"Removed {len(images_to_remove)} images without a detection from the cloud."
        )

    if opt.stage == "after_pipeline":
        # For a container, delete all content from a certain date.
        cnames = ["blurred", "detections"]

        for cname in cnames:
            all_data = saClient.list_container_content(
                cname=cname,
                blob_prefix=start_date_dag,
            )
            if all_data:
                saClient.delete_blobs(
                    cname=cname,
                    blob_names=all_data,
                )
                print(f"Removed {len(all_data)} blobs from container {cname}.")
            else:
                print(
                    f"No blobs found in container {cname} for date {start_date_dag_ymd}."
                )
