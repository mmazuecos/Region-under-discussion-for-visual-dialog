# GuessWhat?! Effectiveness 
Repository for computing effectiveness on model-generated dilogues.

Based on the code for [this paper](https://arxiv.org/abs/1809.03408). To run
this code refer to [this file](README_prev.md).

You can compute effectivenes by running the script run.sh:

`bash run.sh`

# Run Visualizer

To run the visualizer of certain objects run jupyter and open Viz.ipynb.
You will need to enable ipywidgets. To do so, you can run the following lines:

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

Also, you will need the data stored in tha **data/** directory.
