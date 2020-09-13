# coding: utf-8
import argparse
import colorsys
import cubes
import datetime
import imgaug
import os
import random
import shutil
import skimage.draw
import sys
import time
import torch
import urllib.request
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np

# Don't show warnings when running code as the version of
# Mask R-CNN implementation is deprecated
if not sys.warnoptions:
    warnings.simplefilter('ignore')

from .build_mask import Build_mask
from unet.utils import evaluate_amodal
from unet.model import Unet
from mrcnn.config import Config
from mrcnn import visualize as viz
from mrcnn import utils
from mrcnn import model as modellib
from skimage.color import gray2rgb

# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mrcnn', 'mask_rcnn_coco.h5')

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'mrcnn', 'logs')
DEFAULT_DATASET_YEAR = '2017'

# GPU for torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


############################################################
#  Configurations
############################################################
class GrapeConfig(Config):
    # Give the configuration a recognaizable name
    NAME = 'grape'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2
    # grape has 2 classes(fruits+stem)


############################################################
#  Dataset
############################################################
class GrapeDataset(utils.Dataset):
    def load_grape(self, dataset_dir, subset,
                   year=DEFAULT_DATASET_YEAR,
                   class_ids=None,
                   return_grape=False,
                   auto_download=False):
        """Load a subset of the COCO dataset.

        dataset_dir: Root directory of the COCO dataset.
        subset: Subset to load (train, val, minival, valminusminival).
        year: Year of the dataset to load (2014, 2017).
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images
            and annotations.
        """

        # if subset == 'minival' or subset == 'valminusminival':
        #     subset = 'val'
        image_dir = '{}'.format(dataset_dir)
        # txtは通し番号　各画像ファイル名　高さ　幅を持っているようにする
        with open('samples/coco/{}_list.txt'.format(subset), 'r') as f:
            grape = []
            i = 0
            for row in f:
                grape.append(row.split())
                # [i][j]に通し番号i番目に[i][1]に画像ファイル名　[i][2]に高さ　[i][3]に幅が入る
                i += 1

        # Load all classes or a subset
        if not class_ids:
            # All classes
            # 1: stem; 2: grape
            class_ids = [1, 2]

        # Add classes
        grapeclassname = ['stem', 'fruits']
        for i in class_ids:
            self.add_class('grape', i, grapeclassname[i-1])
            # class情報の追加

        # Add images
        for g in grape:
            self.add_image('grape', image_id=i,
                           path=os.path.join(image_dir, g[1]),
                           width=g[3], height=g[2],
                           annotations=[])
            # annotationsはload_maskでのみ使用なので本コードでは不要
        if return_grape:
            return grape

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # If not a grape image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "grape":
            return super(GrapeDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        mask_name, exe = self.image_info[image_id]["path"].split('_rgb')
        mask_name = mask_name+"_label.png"
        # print(mask_name)
        instance_masks, class_ids = Build_mask(mask_name)

        # Pack instance masks into an array

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(GrapeDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons,
            uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
# Evaluation
############################################################
def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i/N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=1):
    # alphaの部分を変更するとマスク画像の透過度を調整できる
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha*color[c] * 255,
                                  image[:, :, c]).astype(np.uint8)
    return image


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def color_splash(image, mask):
    """Apply color splash effect.
    image:RGB image [height , width, 3]
    mask:instance segmentation mask [height,widht,instance count]

    Returns result image.
    """

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image))*255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model,
                            image_path=None,
                            video_path=None,
                            live_path=None):
    assert image_path or video_path or live_path
    # この部分でも画像のセグメンテーションの実行は可能だが，調整が面倒だったので実際には使用していない
    # Image or video
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        print(image.shape)
        # if len(image.shape)<3:
        #     image = gray2rgb(image)
        # else:
        #     image = rgba2rgb(image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    # 通信をしながらセグメンテーションをする
    elif live_path:
        import cv2
        import time
        import multiprocessing
        import struct
        import zmq
        import tensorflow as tf
        mydevice = '/gpu:0'
        colors = random_colors(200)
        q13 = multiprocessing.Queue()
        q32 = multiprocessing.Queue()

        def receive():
            conn_str = "tcp://*:61555"
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REP)
            sock.bind(conn_str)
            for num in range(329):
                # print("受信待機")

                # ヘッダーの受信（受信できるまで待つ）
                byte_rows, byte_cols, byte_mType, data = sock.recv_multipart()
                sock.send(np.array([3]))
                # ヘッダーを解釈する
                rows = struct.unpack('i', byte_rows)
                cols = struct.unpack('i', byte_cols)
                mat_type = struct.unpack('i', byte_mType)

                # 実際に受信する
                if mat_type[0] == 0:
                    # Gray Scale
                    image = np.frombuffer(data, dtype=np.uint8).reshape(
                        (rows[0], cols[0]))
                else:
                    # BGR Color
                    image = np.frombuffer(data, dtype=np.uint8).reshape(
                        (rows[0], cols[0], 3))
                # print("func1")
                print("受信")
                # lock.acquire()
                q13.put(image)
                # lock.release()
                time.sleep(2)

        def seg():
            roop_c = 0
            with tf.device(mydevice):
                for num in range(329):
                    roop_c = roop_c+1
                    # lock.acquire()
                    image = q13.get()
                    # lock.release()
                    image = image[..., ::-1]
                    r = model.detect([image], verbose=0)[0]
                    N = r['rois'].shape[0]
                    splash = np.zeros_like(image.astype(np.uint8))

                    for i in range(N):
                        color = colors[i]
                        if r['class_ids'][i] == 1:
                            continue

                        else:
                            splash = apply_mask(
                                splash, r['masks'][:, :, i], color)
                    splash = splash[..., ::-1]
                    print(roop_c)
                    q32.put(splash)

        def send():
            conn_str = "tcp://localhost:63000"
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.connect(conn_str)

            for num in range(329):
                # print("send待機")
                # publish
                image2 = q32.get()

                height, width = image2.shape[:2]
                ndim = image2.ndim

                # pythonからは4つのデータを配列として順次送る
                data = [np.array([height]), np.array([width]),
                        np.array([ndim]), image2.data]
                sock.send_multipart(data)
                sock.recv()
                print("送信")

        print("start")
        cluster = tf.train.ClusterSpec({'local': ['localhost:62222']})
        server_host = tf.train.Server(cluster, job_name='local', task_index=0)

        sess = tf.Session(server_host.target)
        # lock = multiprocessing.Lock()
        p1 = multiprocessing.Process(target=receive)
        p1.start()
        p3 = multiprocessing.Process(target=send)
        p3.start()
        sess.run(seg())

        # p1.join()
        # p2.join()
        p3.join()

        # 動画を入力としたセグメンテーション
    elif video_path:
        import cv2
        # Video capture
        print(video_path)
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # Define codec and create video writer
        file_name = "{}_{:%Y%m%dT%H%M%S}.mp4".format(
            video_path, datetime.datetime.now())
        vwriter = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(* 'MP4V'), fps, (480, 480))
        count = 0
        success = True
        colors = random_colors(200)
        while success:
            # Read next image
            #             print("read")
            #             start =time.time()
            success, image = vcapture.read()
            if success:
                # cubemap作成箇所
                image = np.ravel(image)
                image = np.array(cubes.cubemap_dest0(image))
                image = np.reshape(image, (480, 480, 3))
                # cubemap作成終了
                # OpenCV returns images as BGR,convert to RGB
                image = image[..., ::-1]
                # Detect objects
                # time_detect = time.time()
                r = model.detect([image], verbose=0)[0]
                # ここでセグメンテーションをし，以下で出力するための処理をしている

                # detect_time = time.time() - time_detect
                N = r['rois'].shape[0]
                # colors = random_colors(N)
                # Color splash
                # splash=image.astype(np.uint8).copy()
                splash = np.zeros(image.shape, np.uint8)
                for i in range(N):
                    color = colors[i]
                    # class_id=1は軸の部分なので，精度が悪いため出力しないようにしている
                    if r['class_ids'][i] == 1:
                        continue
                    else:
                        splash = apply_mask(splash, r['masks'][:, :, i], color)
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # elapsed_time = time.time() -start

                # Add image to video writer
                vwriter.write(splash)
                # cv2.imwrite('./results/frame/frame'+str(count)+'.png',splash)
                print("frame: ", count)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


# 使わない
def build_grape_results(dataset, image_ids, rois, class_ids, scores, masks):
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        for i in range(rois, shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            results = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


# 使わない
def evaluate_grape(model, dataset, grape,
                   eval_type="bbox",
                   limit=0,
                   image_ids=None):
    image_ids = image_ids or dataset.image_ids

    if limit:
        image_ids = image_ids[:limit]

    grape_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        image = dataset.load_image(image_id)

        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        image_results = build_grape_results(dataset, grape_image_ids[i:i+1],
                                            r["rois"], r["class_ids"],
                                            r["scores"],
                                            r["masks"].astype(np.uint8))
        results.extend(image_results)

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on grape.')
    parser.add_argument('command', metavar='<command>',
                        help="'train' or 'evaluate' on grape")
    parser.add_argument('--year', required=False, default=DEFAULT_DATASET_YEAR,
                        metavar='<year>', help='Year of the grape dataset')
    parser.add_argument('--model', required=True, default='last',
                        metavar='/path/to/weights.h5',
                        help='Path to .h5 weights file.')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar='/path/to/logs/',
                        help='Logs and checkpoints directory.')
    parser.add_argument('--limit', required=False,
                        default=434, metavar='<image count>',
                        help='Images to use for evaluation.')
    parser.add_argument('--download', required=False, action='store_true',
                        help='Automatically download and unzip MS-COCO files.')
    parser.add_argument('--amodal', required=False, action='store_true',
                        help='Apply amodal completion or not.')
    parser.add_argument('--amodal_model', required=False, default='baseline',
                        help='Model used for amodal completion.')
    # Mutual exclusion for those 4 options, only one can be chose at a time
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dataset', metavar='/path/to/grape/',
                      help='Directory of the grape dataset.')
    mode.add_argument('--image', metavar='path or URL to image',
                      help='Image to apply the color splash effect on.')
    mode.add_argument('--video', metavar='path or URL to video',
                      help='Video to apply the color splash effect on.')
    mode.add_argument('--live', metavar='live streaming mode',
                      help='Live streaming mode.')
    args = parser.parse_args()

    # Make sure to specify a model to use for amodal completion when
    # using the amodal option (for amodal completion)
    if args.amodal:
        if args.amodal_model is None:
            raise argparse.ArgumentError(
                args.amodal_model,
                '--amodal_model requires a value if --amodal option is passed'
            )

    # print('Command:', args.command)
    # print('Model:', args.model)
    # print('Dataset:', args.dataset)
    # print('Year:', args.year)
    # print('Logs:', args.logs)
    # print('Auto Download:', args.download)

    # Configurations
    if args.command == 'train':
        config = GrapeConfig()
    else:
        class InferenceConfig(GrapeConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    # config.display()

    # Create model
    if args.command == 'train':
        model = modellib.MaskRCNN(
            mode='training', config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(
            mode='inference', config=config, model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == 'coco':
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == 'last':
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == 'imagenet':
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print('Loading Mask R-CNN weights:', model_path)
    if args.model.lower() == 'coco':
        model.load_weights(model_path, by_name=True,
                           exclude=[
                               'mrcnn_class_logits',
                               'mrcnn_bbox_fc',
                               'mrcnn_bbox',
                               'mrcnn_mask'
                           ])
    else:
        model.load_weights(model_path, by_name=True)

    if args.amodal:
        # Absolute path to the best model file
        amodal_model_path = os.path.abspath(
            os.path.join(
                'unet',
                args.amodal_model+'.pth'
            )
        )
        print('Loading amodal weights:', amodal_model_path)
        # Create a model with same parameters as the first one
        amodal_model = Unet(nb_classes=1,
                            experiment='',
                            device=device)

        # Load the best model and put it on GPU
        amodal_model.load_state_dict(
            torch.load(
                amodal_model_path,
                map_location=device
            )
        )
        amodal_model.to(device)

    # Train or evaluate
    if args.command == 'train':
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = GrapeDataset()
        dataset_train.load_grape(
            args.dataset, 'train', year=args.year, auto_download=args.download)

        dataset_train.prepare()

        # Validation dataset
        dataset_val = GrapeDataset()
        dataset_val.load_grape(args.dataset, 'minival',
                               year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # ***This training schedule is an example. Update to your needs ***
        # Training - Stage 1
        # この辺りでepochsを変更するとエポック数が変更できる．
        print('Training network heads')
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print('Fine tune Resnet stage 4 and up')
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print('Fine tune all layers')
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=30,
                    layers='all',
                    augmentation=augmentation)
    elif args.command == 'evaluate':
        if args.video:
            detect_and_color_splash(
                model, image_path=args.image, video_path=args.video)
        elif args.live:
            detect_and_color_splash(model, live_path=True)
        else:
            if args.dataset:
                dataset = GrapeDataset()
                dataset.load_grape(args.dataset, 'val',
                                   year=args.year,
                                   return_grape=True,
                                   auto_download=args.download)
                dataset.prepare()
                data = dataset.image_ids
                grape_image_paths = [
                    dataset.image_info[id]['path'] for id in data]
            elif args.image:
                data = [args.image]

            modal_running_precision = 0
            modal_running_recall = 0
            modal_running_fscore = 0
            modal_running_accuracy = 0
            modal_running_err = 0
            amodal_running_precision = 0
            amodal_running_recall = 0
            amodal_running_fscore = 0
            amodal_running_accuracy = 0
            amodal_running_err = 0
            nb_false_pred = 0

            # Validation dataset
            for idx, img in enumerate(data):
                if args.dataset:
                    image = dataset.load_image(img)
                    caption = grape_image_paths[img].split('/')[-1]
                else:
                    image = skimage.io.imread(img)
                    image = skimage.color.gray2rgb(image)
                    caption = args.image.split('/')[-1]

                print('Now processing image: {}'.format(caption))
                t = time.time()
                results = model.detect([image], verbose=0)
                # print('Instance segmentation time: {}s'.format(time.time()-t))

                ax = get_ax(1)
                r = results[0]
                caption, ext = os.path.splitext(caption)

                if args.amodal:
                    amodal_caption = caption + '_gen'
                    stem_idx = np.argwhere(r['class_ids'] == 1).flatten()[0]
                    stem_mask = r['masks'][:, :, stem_idx]
                    masks = np.delete(r['masks'], stem_idx, axis=2)
                    masks = torch.from_numpy(masks).float()
                    masks = masks.permute((2, 0, 1)).unsqueeze(1)

                    # Perform amodal completion
                    t = time.time()
                    amodal_masks = amodal_model.predict(masks, threshold=0.95)
                    # print('Amodal completion time: {}s'.format(time.time()-t))

                    t = time.time()
                    metrics, wrong_masks = evaluate_amodal(
                        masks=masks.squeeze(),
                        amodal_masks=amodal_masks.squeeze(),
                        caption=caption
                    )
                    # print('Compute metrics time: {}s'.format(time.time() - t))
                    modal_running_precision += metrics['modal']['precision']
                    modal_running_recall += metrics['modal']['recall']
                    modal_running_fscore += metrics['modal']['f-score']
                    modal_running_accuracy += metrics['modal']['accuracy']
                    modal_running_err += metrics['modal']['center_err']
                    amodal_running_precision += metrics['amodal']['precision']
                    amodal_running_recall += metrics['amodal']['recall']
                    amodal_running_fscore += metrics['amodal']['f-score']
                    amodal_running_accuracy += metrics['amodal']['accuracy']
                    amodal_running_err += metrics['amodal']['center_err']
                    nb_false_pred += len(wrong_masks)

                    # import pudb; pudb.set_trace()
                    amodal_masks = amodal_masks.squeeze().permute(
                        (1, 2, 0)).numpy().astype(bool)
                    amodal_masks = np.delete(amodal_masks, wrong_masks, axis=2)
                    r['masks'] = np.delete(r['masks'], wrong_masks, axis=2)
                    r['rois'] = np.delete(r['rois'], wrong_masks, axis=1)
                    r['class_ids'] = np.delete(
                        r['class_ids'], wrong_masks, axis=0)
                    r['scores'] = np.delete(r['scores'], wrong_masks, axis=0)
                    stem_idx = np.argwhere(r['class_ids'] == 1).flatten()[0]
                    r['masks'] = np.insert(
                        amodal_masks, stem_idx, stem_mask, axis=2)
                # viz.display_instances(image=image,
                #                       boxes=r['rois'],
                #                       masks=r['masks'],
                #                       class_ids=r['class_ids'],
                #                       class_names=['bg', 'stem', 'fruit'],
                #                       scores=r['scores'],
                #                       ax=ax, title=amodal_caption)
            modal_prec = modal_running_precision/(idx+1)
            modal_recall = modal_running_recall/(idx+1)
            modal_fscore = modal_running_fscore/(idx+1)
            modal_acc = modal_running_accuracy/(idx+1)
            modal_err = modal_running_err/(idx+1)
            amodal_prec = amodal_running_precision/(idx+1)
            amodal_recall = amodal_running_recall/(idx+1)
            amodal_fscore = amodal_running_fscore/(idx+1)
            amodal_acc = amodal_running_accuracy/(idx+1)
            amodal_err = amodal_running_err/(idx+1)

            # Print results of amodal completion
            print('Number of wrong predictions:', nb_false_pred)
            print('Without amodal completion:\n')
            print('Precision:', modal_prec)
            print('Recall:', modal_recall)
            print('F-Score:', modal_fscore)
            print('Accuracy:', modal_acc)
            print('Center error:', modal_err)
            print('--------------------------------------')
            print('With amodal completion:\n')
            sign = '+' if amodal_prec - modal_prec > 0 else '-'
            print('Precision:', amodal_prec,
                  '({}{})'.format(sign, np.abs(amodal_prec - modal_prec)))
            sign = '+' if amodal_recall - modal_recall > 0 else '-'
            print('Recall:', amodal_recall,
                  '({}{})'.format(sign, np.abs(amodal_recall - modal_recall)))
            sign = '+' if amodal_fscore - modal_fscore > 0 else '-'
            print('F-Score:', amodal_fscore,
                  '({}{})'.format(sign, np.abs(amodal_fscore - modal_fscore)))
            sign = '+' if amodal_acc - modal_acc > 0 else '-'
            print('Accuracy:', amodal_acc,
                  '({}{})'.format(sign, np.abs(amodal_acc - modal_acc)))
            sign = '+' if amodal_err - modal_err > 0 else '-'
            print('Center error:', amodal_err,
                  '({}{})'.format(sign, np.abs(amodal_err - modal_err)))
    else:
        print("'{}' is not recognized.\n",
              "Use 'train' or 'evaluate'".format(args.command))
