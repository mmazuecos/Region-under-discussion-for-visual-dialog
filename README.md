# Region-under-discussion-for-visual-dialog

You should have available three things on the root directory for this repository:
1. QCS: directory with QCS code
2. CMO: directory with CMO code
3. REAMDE.md: this file.

In order to run these experiments you need the following data and features:

+ [GuessWhat?! dataset](https://drive.google.com/file/d/1JiJIV_Ve65SHriU8veTtLVWmlM-Nu6pi/view?usp=sharing)
+ MS COCO images ([train](images.cocodataset.org/zips/train2014.zip) and [validation](images.cocodataset.org/zips/val2014.zip) from 2014).
+ The [visual features](https://drive.google.com/file/d/1t1PoKWkrDoKlQwJehtG2mHiuJ5B9-Al2/view?usp=sharing) for the target objects

## QCS
To run experiments, edit the config/Oracle/config.json file or create one.

### Semantic parse
```
$ python3 -m scripts.parse
```

### Run experiment
Modify the run parameters in pipeline.sh

```
# ======================================
# Configure the experiment here:
CONFIG='config/Oracle/config.json'
EXP_NAME='Oracle_exp'
BIN_NAME='_experiment_'
GPU_ID='0'
SPLIT='test'
# ======================================
```
then run
```
$ sh pipeline.sh
```

### 

## CMO

We assume you have downloaded MS-COCO and GuessWhat?!. You'll also need to install [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention).

#### Compute features for MSCOCO

```sh
$ python3 fasterrcnn_mscoco.py \
    --coco-root=${COCO_ROOT} \
    --py-bottom-up-attention-path=${PY_BOTTOM_UP_ATTENTION_PATH} \
    --num-objects=36
```

The script will save one .npy file per image using the same tree structure as in the dataset. You can read the data as follows:

```python
import numpy as np
fname = "fasterrcnn/mscoco_num-objects_36/val2014/COCO_val2014_000000000042.npy"
data = np.load(fname, allow_pickle=True).item()
```

where:

 - data["image_size"]: tuple of ints. Image width and height (in that order)
 - data["scores"]: np.array of size (NUM_OBJECTS,). Detection scores sorted in descening order
 - data["classes"]: list of str. class names for each of the NUM_OBJECTS objects
 - data["boxes"]: np.array of size (NUM_OBJECTS, 4). Bounding boxes in the format (x, y, w, h)
 - data["features"]: np.array of size (NUM_OBJECTS, 2048). Feature vectors for each box

```
python3 -W ignore trainval.py --guesswhat-root=guesswhat_with_focus_final --max-length=32 --batch-size=32 --epochs=5 --coco-data-root=data --oracle-targets-root=resnet152 --focus-mode=none --num-threads=4 --num-workers=2 --suffix=butd-resnet152 --marker=none
```
were focus-mode can be either none, relative, restriction, zeros and random.
