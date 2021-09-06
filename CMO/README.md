# LXMERT Oracle

## Preliminaries

We assume you have downloaded MS-COCO and GuessWhat?!. You'll also need to install [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention).

### Compute features for MSCOCO

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

### Compute features for GuessWhat?! (target objects)

```sh
$ for set in train valid test; do \
    python3 fasterrcnn_guesswhat_oracle.py \
      --guesswhat-root=${GUESSWHAT_ROOT} --set=${set} \
      --coco-root=${COCO_ROOT} \
      --py-bottom-up-attention-path=${PY_BOTTOM_UP_ATTENTION_PATH}; \
  done
```

This will save three .npy, one for each set in the dataset. You can read the data as before. Each file contains a dict whose keys are the game IDs (as strings). Values for each key follow the same format as above with NUM_OBJECTS=1.

## Train baseline model

```sh
python3 baseline.py \
  --coco-data-root=fasterrcnn/mscoco_num-objects_36 \
  --oracle-targets-root=fasterrcnn/
  --guesswhat-root=${GUESSWHAT_ROOT}
  --learning-rate=1e-5 \
  --batch-size=32 \
  --epochs=20
```

training progress and models will be saved at ./cache/. You can monitor the training process with tensorboard.

# Troubleshooting

* [py-bottom-up-attention] compilation error with latest versions of pytorch ([issue](https://github.com/conansherry/detectron2/issues/12))
Edit deform_conv_cuda.cu and deform_conv.h at detectron2/detectron2/layers/csrc/deformable/ and replace all occurrences of “AT_CHECK” with “TORCH_CHECK”.
* [py-bottom-up-attention] compile w/ CUDA 11 support
  ```sh
  $ export CUDA_PATH=/opt/cuda/11.0/   # jupiterace, nabucodonosor2
  $ python3.8 setup.py build develop --user
  $ python3.8 -m pip install . --user
  ```
