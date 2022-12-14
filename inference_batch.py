import os
import cv2
from tqdm import tqdm
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


class ContainerDataset(Dataset):
    def __init__(self,img_names,cfg=None):
        self.img_names = img_names
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        ) # TODO validate the input dims
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


def create_csv(result_save_dir):
    f = open(result_save_dir+"/detection_result_batch.csv", "w", newline="")
    csv_writer = csv.writer(f, delimiter=";")
    csv_writer.writerow(["file_name", "bbox", "score"])

    return f, csv_writer


if __name__ == "__main__":
    result_save_dir = "jm"
    data_path = "data_sample/test/"

    f, csv_writer = create_csv(result_save_dir)

    cfg = get_cfg()
    cfg.merge_from_file("configs/container_detection.yaml")
    cfg.DATALOADER.NUM_WORKERS = 1
    BATCH_SIZE = 1

    model = build_model(cfg)
    DetectionCheckpointer(model).load("weights/model_final2.pth")
    model.train(False)

    image_names = [file_name for file_name in glob.glob(f"{data_path}*.jpg")]
    dataset = ContainerDataset(img_names=image_names, cfg=cfg)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             pin_memory=True)

    num = 0
    all_rows = []
    with torch.no_grad():
        for (imgs, img_names, shapes) in tqdm(data_loader):  # TODO dont use tqdm in production

            inputs = []

            for i, img_tensor in enumerate(imgs):
                inputs.append({"image": img_tensor, "height": shapes[0][i], "width": shapes[1][i]})

            all_outputs = model(inputs)

            for j, outputs in enumerate(all_outputs):
                if len(outputs["instances"]) == 0:  ### no predicted objects ###
                    #all_rows.append([img_names[j], '', '', '', '', 0])
                    continue

                for k in range(len(outputs["instances"])):
                    xmin = round(outputs["instances"].pred_boxes.tensor[k][0].item(), 1)
                    ymin = round(outputs["instances"].pred_boxes.tensor[k][1].item(), 1)
                    xmax = round(outputs["instances"].pred_boxes.tensor[k][2].item(), 1)
                    ymax = round(outputs["instances"].pred_boxes.tensor[k][3].item(), 1)
                    bbox = {xmin, ymin, xmax, ymax}

                    score = round(outputs["instances"].scores[k].item(), 3)

                    all_rows.append([os.path.basename(img_names[j]), bbox, score])

                num += 1

    for row in all_rows:
        csv_writer.writerow(row)

    f.close()
    print("Detect %d frames with objects in haul %s"%(num, data_path))
