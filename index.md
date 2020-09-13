# [Amodal Completion for Estimating Centers of Grapes](https://drive.google.com/file/d/1RlYYrtNAfvKLiXu55iUw62KNdhcr1ZMb/view?usp=sharing)

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) and [U-Net](https://arxiv.org/abs/1505.04597) on Python 3, Keras, TensorFlow and PyTorch. The framework performs instance segmentation, followed by amodal semantic segmentation on the masks from instance segmentation. We used the implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) from [Matterport](https://github.com/matterport/Mask_RCNN). Please refer to the official [github page](https://github.com/matterport/Mask_RCNN) for more information on the [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation. (Note we use an outdated implementation in our current repository.)

You can also check my paper [here](https://drive.google.com/file/d/1RlYYrtNAfvKLiXu55iUw62KNdhcr1ZMb/view?usp=sharing)!

![amodal_completion_results](https://drive.google.com/uc?export=view&id=1ytwCsbsm8kt4eGvKokuC3ePccrVT6M_p)

# Framework
Before running any of the commands below, make sure to follow the instructions in the [Installation section](#installation).

Running the grape code will store all the results images in the `results` folder.

If you want to crop the all resulting images in the folder, you can run the `crop_results.py` script to only keep the image and discard the blank around the image saved from Matplotlib:
```bash
python crop_results.py
```

## 1. Mask R-CNN for instance segmentation
Possible argument to pass to the command:
* Required arguments:
    * command: 'train' or 'evaluate' on grape
    * --model: Path to .h5 weights file
    * Only one of the following option can be pass at one time:
        * --dataset: Directory of the grape dataset
        * --image: Image to apply the color splash effect on
        * --video: Video to apply the color splash effect on
        * --live: Live streaming mode
* Optional arguments:
    * --year: Year of the grape dataset (default 2017)
    * --logs: Logs and checkpoints directory (default=./mrcnn/logs/)
    * --limit: Images to use for evaluation (default=434)
    * --download: Automatically download and unzip MS-COCO files (default False)

### Training on grape dataset
```bash
python -m samples.coco.grape train --model=coco --dataset=./data/fastgrape/grape/train/
```

### Evaluating on grape dataset
```bash
python -m samples.coco.grape evaluate --model=last --dataset=./data/fastgrape/grape/val/
```

### Evaluating on one image
```bash
python -m samples.coco.grape evaluate --model=last --image=./data/fastgrape/grape/val/fuji/before_fuji234_05_rgb.JPG
```

For more details on the use of Mask R-CNN for instance segmentation only, please refer to the [instructions](./mrcnn/instructions.pdf) written by 尾形 亮輔.

## 2. U-Net for amodal completion
Possible argument to pass to the command:
* Required arguments:
    * --training / -train: Set training mode (cannot be passed if -eval is already passed)
    * --evaluate / -eval: Set evaluation mode (cannot be passed if -train is already passed)
    * --debug / -d: Set the experiment's name to 'debug' for debugging (cannot be passed if -exp is already passed)
    * --experiment / -exp: Extension for the name of the experiment (cannot be passed if -debug is already passed)
* Optional arguments:
    * --batch / -b: Specify the batch size (default 4)
    * --workers / -w: Number of workers for DataLoader function (default 4)
    * --learning_rate / -lr: Learning rate for the network (default 0.001)
    * --epoch / -e: Set the number of epochs (> 5) (default 50)
    * --threshold / -thresh: Threshold for evaluation (default 0.5)
    * --augmentation / -aug: Do data augmentation (default None)

### Training on synthetic grape dataset
```bash
python -m unet.main -train -exp [name_of_your_experiment]
```
The experiments will be saved in the output folder, created in unet folder under the name:
```
Unet-{mmddyy}-lr-{learning rate value}-b{batch size value}_{name of your experiment}
```
Once trained, your folder should looks like something like this:
```
.
├── data/
├── mrcnn/
├── samples/
└── unet
    ├── ...
    └── output
        ├── ...
        └── Unet-{date}-lr-{learning rate value}-b{batch size value}_{name of your experiment}
            ├── eval
            └── train
                ├── logs
                └── models
```

During the training, the logs for tensorboard are stored in the `logs` subfolder, and the models in the `models` subfolder. The best model is saved under the name `best_model.pth`, which can be renamed (but not the extension) in the [`model.py`](./unet/model.py) file (line 357). Plus, every one fifth of the total number of epoch, the model is save, in case of a crash or power outage.

### Evaluating on synthetic grape dataset
Here you must specitfy the path to the folder of your experiment from the root of the directory:
```bash
python -m unet.main -eval -exp ./unet/output/your_experiment_folder/
```

The results will be stored as images in the `eval` subfolder in the experiment folder (see above for folder structure).

### Data augmentation for amodal completion
With the parameter `--augmentation` or `-aug`, you can enable data augmentation during the training of amodal completion model. All the implemented augmentation are grouped in the [`data_aug.py`](./unet/data_aug.py) file. The augmentation are seperated into classes and all the augmentation are passed in a list to our personalized data augmentation class, which calls the transformations and apply shuffling or a random choice of augmentation. All the augmentations used in our project are from torchvision.

#### Create your own augmentation
You can add your own augmentation and use data augmentation operations from other packages. However, as we perform online data augmentation to make sure the format is preserved when loading data, you should create a class which performs the application of the desired augmentation. To create your own class, you can refer to this simple template below:

```python
import torchvision.transforms.functional as tff

class MyAugmentation(object):

    def __call__(self, sample):
        sample['image'] = my_augmentation(sample['image'])
        sample['target'] = my_augmentation(sample['target'])
        return sample
```

You have to apply the transformation on the image and its target if your augmentation affects the structure of the image.  As our sample is a python dictionary, it must be done separately.

## 3. Combine Mask R-CNN and U-Net
To perform amodal completion on the results of instance segmentation, you just need to add two more arguments to the commands of Mask R-CNN because they're not set by default:
* --amodal: Apply amodal completion or not
* --amodal_model: Model used for amodal completion (by default, baseline model is used)

Once a model is trained with U-Net, use the `best_model.pth` model from the `unet/output/[experiment_folder]/train/models/` folder to perform amodal completion with Mask R-CNN. The file must be put in the root of unet folder. To make them more explicit, you can also rename them, like we did for our baseline model:

```
.
├── data/
├── mrcnn/
├── samples/
└── unet
    ├── ...
    ├── baseline.pth
    └── your_model.pth
```

Feel free to change the amodal model by the one you trained. You just have to specify the name of the `.pth` file through the `--amodal_model` parameter.

### Perform amodal completion on grape dataset
```bash
python -m samples.coco.grape evaluate --model=last --dataset=./data/fastgrape/grape/val/ --amodal --amodal_model=baseline
```

### Perform amodal completion on one grape image
```bash
python -m samples.coco.grape evaluate --model=last --image=/path/to/image --amodal --amodal_model=baseline
```

## Requirements
Python 3.4+, TensorFlow 1.3, Keras 2.0.8, Torch 1.4 and other common packages listed in `requirements.txt`.

## Installation
1. (Optional) If not existing, create a virtual environment in the directory and activate it:
    ```bash
    virtualenv -p python3 my_venv
    source venv_amodalCompletion/bin/activate
	```
	Otherwise just activate the existing virtual environment:
	```bash
    source venv_amodalCompletion/bin/activate
    ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Clone this repository.
4. Create empty `data` and `results` folders in the root of the repository. (All the data can be found on the IMP server to the following path: `/disk020/usrs/julian/amodal_completion/data/`.)
5. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases) and save the file in the `mrcnn` directory (only if you want to train Mask-RCNN from scratch again). (**Note**: Make sure to also retrieve the `logs` folder of Mask R-CNN here: `/disk020/usrs/julian/amodal/completion/mrcnn/`.)
6. If not already existing, add the grape and synthetic grape datasets to the data folder and run the following command:
    ```bash
    python ./unet/data_to_json.py -f ./data/synthethic_grape/
    ```

At the end of the installation, the structure of your directory should be something like this:
```
.
├── data
│   ├── fastgrape/
│   └── synthetic_grape/
│       ├── grapes/
│       ├── imgs/
│       └── fname_dataset.json
├── mrcnn
|   ├── ...
|   ├── logs
|   └── mask_rcnn_coco.h5
├── samples
│   └── coco/
└── unet/
```
The last command will create the `fname_dataset.json` file, used to generate train, val and test set for amodal completion on the synthetic dataset.
