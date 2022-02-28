# README

This repository contains source code and data of my thesis.

## Dataset

We provide our pre-processed data in the `data` folder.

### Twitter15 & Twitter16
 
The raw Twitter dataset can be downloaded from [raw-Twitter(1)](https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0) 
and from [raw-Twitter(2)](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0&file_subpath=%2Frumor_detection_acl2017).


### Pheme

The raw Pheme dataset can be downloaded from [raw-Pheme](https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619).


## Project Structure

``` txt

├── README.md
├── main.py
├── data
│   ├── README.md
│   ├── Pheme
│   │   ├── Phemegraph
│   │   ├── Phemelink
│   │   └── PhemeText
│   └── Twitter
│        ├── Twitter15
│        ├── Twitter15graph
│        ├── Twitter16
│        └── Twitter16graph
├── graph
├── model
│   ├── bigcn.py
│   ├── gcn.py
│   ├── revised_bigcn.py
│   └── revised_gcn.py
├── process
│   ├── dataset.py
│   ├── get_graph.py
│   ├── load_data.py
│   └── rand5fold.py
├── requirements.txt
├── results
│   └── README.md
└── tools
    ├── earlystopping.py
    ├── evalutate.py
    └── pheme_label_preprocess.py

```


## Dependencies

See `requirements.txt` file.


## Experiment

Click [here](./results) to see our experiment results.

```
# Generate graph data and store in /data/Twitter/Twitter15graph
python ./process/get_graph.py Twitter15
# Generate graph data and store in /data/Twitter/Twitter16graph
python ./process/get_graph.py Twitter16
# Generate graph data and store in /data/Pheme/Phemegraph
python ./process/get_graph.py Pheme


# Run experiment
python main.py <dataset_name> <iteration_times> <model_name>

* Attention: Twitter dataset cannot be used for revised version model.
```


## Notice

When you use Pycharm to run the program and meet the **out of memory** problem, please change your IDE memory allocation.

Useful link:

1. [Understanding IDE memory allocation](https://intellij-support.jetbrains.com/hc/en-us/articles/360018776919-Understanding-IDE-memory-allocation-)
2. [Pycharm: Advanced configuration](https://www.jetbrains.com/help/pycharm/tuning-the-ide.html)
3. [How to increase Memory for Pycharm : 3 quick Steps only](https://www.datasciencelearner.com/how-to-increase-memory-for-pycharm/)




