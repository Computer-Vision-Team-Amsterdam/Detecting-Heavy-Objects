import os
import cv2
import csv
import glob
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import argparse
from utils.azure_storage import StorageAzureClient
from utils.date import get_start_date
import upload_to_postgres
from psycopg2.extras import execute_values


class ContainerDataset(Dataset):
    def __init__(self,img_names,cfg=None):
        self.img_names = img_names
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )  # TODO validate the input dims
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __getitem__(self, index):
        img = cv2.imread(self.img_names[index])
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            img = img[:, :, ::-1]
        image = self.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        img_name = self.img_names[index]

        return image, img_name, img.shape[:2]

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", type=str, help="Processing date in the format %Y-%m-%d %H:%M:%S.%f"
    )
    parser.add_argument(
        "--device", type=str, help="Processing on CPU or GPU?"
    )
    parser.add_argument(
        "--weights", type=str, help="Trained weights filename"
    )
    opt = parser.parse_args()

    start_date_dag, start_date_dag_ymd = get_start_date(opt.date)

    input_path = Path("images", start_date_dag_ymd)
    if not input_path.exists():
        input_path.mkdir(exist_ok=True, parents=True)

    # Download images from storage account
    saClient = StorageAzureClient(secret_key="data-storage-account-url")
    blobs = saClient.list_container_content(cname="blurred", blob_prefix=start_date_dag)
    for blob in blobs:
        filename = blob.split("/")[-1]  # only get file name, without prefix
        saClient.download_blob(
            cname="blurred",
            blob_name=blob,
            local_file_path=f"{input_path}/{filename}",
        )

    # Make a connection to the database
    conn, cur = upload_to_postgres.connect()

    cfg = get_cfg()
    cfg.merge_from_file("configs/container_detection.yaml")
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.DEVICE = opt.device
    BATCH_SIZE = 1

    model = build_model(cfg)
    DetectionCheckpointer(model).load(opt.weights)
    model.train(False)

    image_names = [file_name for file_name in glob.glob(f"{input_path}/*.jpg")]
    dataset = ContainerDataset(img_names=image_names, cfg=cfg)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             pin_memory=True)

    num = 0
    all_rows = []
    with torch.no_grad():
        for (imgs, img_names, shapes) in data_loader:
            inputs = []
            for i, img_tensor in enumerate(imgs):
                inputs.append({"image": img_tensor, "height": shapes[0][i], "width": shapes[1][i]})

            all_outputs = model(inputs)

            for j, outputs in enumerate(all_outputs):
                if len(outputs["instances"]) == 0:  ### no predicted objects ###
                    continue

                for k in range(len(outputs["instances"])):
                    xmin = round(outputs["instances"].pred_boxes.tensor[k][0].item(), 1)
                    ymin = round(outputs["instances"].pred_boxes.tensor[k][1].item(), 1)
                    xmax = round(outputs["instances"].pred_boxes.tensor[k][2].item(), 1)
                    ymax = round(outputs["instances"].pred_boxes.tensor[k][3].item(), 1)
                    bbox = [xmin, ymin, xmax, ymax]

                    score = round(outputs["instances"].scores[k].item(), 3)

                    all_rows.append([os.path.basename(img_names[j]), score, bbox])

                num += 1

    print("Detect %d frames with objects in haul %s"%(num, input_path))

    if all_rows:
        print("Inserting data into database...")
        table_name = "detections"
        # Get columns
        sql = f"SELECT * FROM {table_name} LIMIT 0"
        cur.execute(sql)
        table_columns = [desc[0] for desc in cur.description]
        table_columns.pop(0)  # Remove the id column
        query = f"INSERT INTO {table_name} ({','.join(table_columns)}) VALUES %s"

        print(all_rows)

        execute_values(cur, query, all_rows)
        conn.commit()
