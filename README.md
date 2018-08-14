# Neural Network Encapsulation



[Hongyang Li](http://www.ee.cuhk.edu.hk/~yangli/), 
[Xiaoyang Guo](https://scholar.google.com/citations?user=CrK4w4UAAAAJ&hl=en), [Bo Dai](http://daibo.info/), *et al*.


The official implementation in Pytorch, paper published in ECCV 2018.

- [arXiv link](https://arxiv.org/abs/1808.03749) (recommended)
- [another](http://www.ee.cuhk.edu.hk/~yangli/paper/eccv18_capsule.pdf) 
(same version as arXiv except with a nicer Google scholar icon)
- [official ECCV proceeding; to appear]() (no appendix)


![capsule](data/capsule.png)

## Overview

- PyTorch `0.3.x` or `0.4.x`; Linux; Python 3.x 
    - I haven't tested on MacOS or Python 2.x, but it won't be a big problem; have a try!
- Provide our own implementation of the original papers, namely 
[dynamic](https://arxiv.org/abs/1710.09829) and [EM](https://openreview.net/pdf?id=HJWLfGWRb) 
routing.
- Datasets: MNIST, CIFAR-10/100, SVHN and a subset of ImageNet.

On a **research** side:

- Analyze the two routing schemes (Dynamic and EM) in original capsule papers.
- Propose an approximation routing workaround to tackle the computational inefficiency, in a supervised manner.
Network elements are still in form of capsules (vector other than scalar).
- That is why we call the network is **encapsulated**.
- Adopt the optimal transport algorithm to make higher and lower capsules align better.





## Grab and Go

The easiest way to run the code **in the terminal**, after 
cloning/downloading this repo is:

    python main.py

If you are more ambitious to play with the parameters
and/or assign the experiment to specific GPUs:

    # gpu_id index
    CUDA_VISIBLE_DEVICES=0,2 \
        python main.py \
            --device_id=0,2 \
            --experiment_name=encapnet_default \
            --dataset=cifar10 \
            --net_config=encapnet_set_OT \
            # other arguments here ...
    
For a full list of arguments, see `option/option.py` file. 
Note **how we launch** the multi-gpu mode above (pass index `0,2` to both environment
variables and arguments). 

## A Deeper Look

#### File Structure
This project is organized in the most common manner:

    | main.py
    |       |
    |       layers/train_val.py
    |               |
    |               layers/network.py               # forward flow control
    |                       |
    |                       -->  model define in net_config.py
    |                       -->  cap_layer.py       # capsule layer submodules; core part
    |                       -->  OT_module.py       # optimal transport unit; core part
    |       data/create_dset.py
    |       option/option.py
    |       utils

Datasets will be automatically downloaded and put under ``data`` folder. Output files (log, model) 
reside in the `--base_save_folder` (default is `result`).
#### Adapting our work to your own task

- To add more structures or change components:
    - write parallel network design in 
the ``if-else`` statement starting from this 
[net_config.py](layers/models/net_config.py#L109) file.

- To add one encapsulated layer with (or not) OT unit in your own network:
    - see code block [here](layers/models/net_config.py#L139-L150) 
    in the  ``net_config.py`` for layer definition and 
    the forward flow [here](layers/network.py#L69-L71) in the ``network.py``.


## Features to come

- [x] Code release for original capsule papers
- [x] Support PyTorch `0.4.x`
- [ ] Supoort `visdom`. Use [visdom](https://github.com/facebookresearch/visdom) 
to visualize training dynamics.
- [ ] h-ImageNet dataset and result
- [ ] Better documentation and performance results


## Citation
Please cite in the following manner if you find it useful in your research:
```
@inproceedings{li2018encapsulation,
  author = {Hongyang Li and Xiaoyang Guo and Bo Dai and Wanli Ouyang and Xiaogang Wang},
  title = {Neural Network Encapsulation},
  booktitle = {ECCV},
  year = {2018}
}
```