# Cloned this commit https://github.com/ultralytics/yolov5/commit/454dae1301abb3fbf4fd1f54d5dc706cc69f8e7e

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path
import time
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, colorstr, cv2,
                           non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

# In the dockerfile we copy a folder from root to this dir and rename it to utils_cvteam
from utils_cvteam.azure_storage import BaseAzureClient, StorageAzureClient
from utils_cvteam.date import get_start_date
azClient = BaseAzureClient()

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        output_folder=Path(ROOT / 'runs/detect'),  # save results to project/name
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        no_save_img=True,  # dont save blurred images
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        date=''  # not used
):
    source = str(source)
    # Directories
    (output_folder / 'labels' if save_txt else output_folder).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    start_time = time.time()

    # Dataloader TODO make batch inference implementation
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, dt = 0, (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(output_folder / p.name)  # im.jpg
            txt_path = str(output_folder / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if no_save_img:
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        blur = im0[y1:y2, x1:x2]
                        blurred = cv2.GaussianBlur(blur, (45,45), 0)
                        im0[y1:y2, x1:x2] = blurred

            # Save results (image with blurs)
            if no_save_img:
                cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or no_save_img:
        s = f"\n{len(list(output_folder.glob('labels/*.txt')))} labels saved to {output_folder / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', output_folder)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    print("--- YOLOv5 blur took %s seconds ---" % (time.time() - start_time))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=f'{os.getcwd()}/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=f'{os.getcwd()}/unblurred', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=f'{os.getcwd()}/data/pano.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--output-folder', type=Path, default=f'{os.getcwd()}/blurred', help='location where blurred images are stored')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[2048, 1024], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--no-save-img', action='store_false', help='don"t save the blurred images')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    parser.add_argument('--date', type=str, help='date for images to be blurred')
    parser.add_argument('--worker-id', type=int, help='worker ID')
    parser.add_argument('--num-workers', type=int, help='number of workers')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


def get_chunk_pano_ids():
    # Download txt file(s) with pano ids that we want to download from CloudVPS
    local_file = f"{opt.worker_id}.txt"
    saClient.download_blob(
        cname="retrieve-images-input",
        blob_name=f"{start_date_dag}/{local_file}",
        local_file_path=local_file,
    )
    with open(local_file, "r") as f:
        pano_ids = [line.rstrip("\n") for line in f]

    if len(pano_ids) < opt.num_workers:
        raise ValueError("Number of workers is larger than items to process. Aborting...")
    print(f"Printing first and last file names from the chunk: {pano_ids[0]} {pano_ids[-1]}")

    return pano_ids


def download_panos():
    # Get all file names of the panoramic images from the storage account
    blobs = saClient.list_container_content(
        cname="unblurred", blob_prefix=start_date_dag
    )

    # Validate if all blobs are available
    if len(set(pano_ids) - set(blobs)) != 0:
        raise ValueError("Not all panoramic images are available in the storage account! Aborting...")

    for blob in pano_ids:
        saClient.download_blob(
            cname="blurred",
            blob_name=f"{start_date_dag}/{blob}",
            local_file_path=f"{opt.output_folder}/{blob}",
        )


if __name__ == "__main__":
    opt = parse_opt()

    start_date_dag, _ = get_start_date(opt.date)

    # update input folder
    opt.source = Path(opt.source, start_date_dag)
    if not opt.source.exists():
        opt.source.mkdir(exist_ok=True, parents=True)

    # update output folder
    opt.output_folder = Path(opt.output_folder, start_date_dag)
    if not opt.output_folder.exists():
        opt.output_folder.mkdir(exist_ok=True, parents=True)

    # download images from storage account
    saClient = StorageAzureClient(secret_key="data-storage-account-url")

    pano_ids = get_chunk_pano_ids()
    download_panos()

    main(opt)

    # upload blurred images to storage account
    for file in os.listdir(f"{opt.output_folder}"):
        saClient.upload_blob(
            cname="blurred",
            blob_name=f"{start_date_dag}/{file}",
            local_file_path=f"{opt.output_folder}/{file}",
        )
