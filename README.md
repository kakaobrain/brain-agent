# Brain Agent
NOTE: This branch is only for NetHack. We do not support training DMLAB agent on this branch.


We haven't applied PopArt, TransformerXL-I, and reconstruction loss for NetHack. 
However, we expect that PopArt may stabilize multi-role training of NetHack and TransformerXL-I may bring the better function of memory.


## Installation
This code has been developed under Python 3.7, Pytorch 1.9.0 and CUDA 11.1.
For NetHack environment, please follow installation instructions from [NLE Github](https://github.com/facebookresearch/nle).

Please run the following command to install the other necessary dependencies.
```bash
pip install -r requirements.txt
```


## Training
The script `train.py` is used for training. Example usage to train our baseline model with GRU-based policy network on 4 nodes having 4 GPUs on each node is
```bash
sleep 120; python -m dist_launch --nnodes=4 --node_rank=0 --nproc_per_node=4 --master_addr=$MASTER_ADDR -m train \ 
cfg=configs/nethack/baseline.yaml train_dir=$TRAIN_DIR experiment=EXPERIMENT_DIR 
sleep 120; python -m dist_launch --nnodes=4 --node_rank=1 --nproc_per_node=4 --master_addr=$MASTER_ADDR -m train \ 
cfg=configs/nethack/baseline.yaml train_dir=$TRAIN_DIR experiment=EXPERIMENT_DIR
sleep 120; python -m dist_launch --nnodes=4 --node_rank=2 --nproc_per_node=4 --master_addr=$MASTER_ADDR -m train \ 
cfg=configs/nethack/baseline.yaml train_dir=$TRAIN_DIR experiment=EXPERIMENT_DIR
sleep 120; python -m dist_launch --nnodes=4 --node_rank=3 --nproc_per_node=4 --master_addr=$MASTER_ADDR -m train \ 
cfg=configs/nethack/baseline.yaml train_dir=$TRAIN_DIR experiment=EXPERIMENT_DIR
```

## Evaluation
### Trained Agent on NLE
We provide the checkpoint of our trained agent on NLE which is used for our submission in Neurips 2022 NetHack Cahllenge. 
Below table shows the avg/median scores of our submission.
Our submission is composed of ensemble of models.
We first trained baseline model, and then we additionally trained role-specific model.
The bold score is actual result of a role of our submission. 
The starred score means that the model is trained for the specific role.


We appended message tokenization and embedding to [nle-sample-factory-baseline](https://github.com/Miffyli/nle-sample-factory-baseline).
We additionally trained role-specific models with encoding items, spells, and pickable items using models using TransformerXL-I and AvgPool.
The configurations of our submission is placed at `configs/nethack/neurips2021_nethack_kakaobrain`.


|  Role | Baseline  | TrXL-I | AvgPool |
|:----------:|:--------:|:----------:|:---------------:|
| All | **1883 / 1188**  | - | - |
| Archeologist | **1424 / 1309**  | - | - |
| Barbarian | **3347 / 2313** | - | - |
| Cave(wo)man | **1874 / 1176***  | - |- |
| Healer | 475 / 316  | **1027 / 918*** | - |
| Knight | **2064 / 1599**  | - | - |
| Monk | **4811 / 3369**  | - | - |
| Priest(ess) | **1257 / 1048** | - | - |
| Ranger | 1566 / 1244  | - | **1759 / 1814*** |
| Rogue | 978 / 868 | - | **1038 / 922*** |
| Samurai | **3107 / 2179** | - | - |
| Tourist | **1227 / 1254*** | - | - |
| Valkyrie | **2419 / 1820** | - | - |
| Wizard | 455 / 256 | **1716 / 1785*** | - |


### Evaluation command
The script `eval.py` is used for evaluation. Example usage to test the provided checkpoint, which is located in $CHECKPOINT_FILE_PATH, is

```bash
python eval.py cfg=configs/nethack/neurips2022/baseline.yaml actor.num_workers=$NUM_WORKERS actor.num_envs_per_worker=2 test.checkpoint=$CHECKPOINT_FILE_PATH 
```
Please set $NUM_WORKERS based on your system specification (number of actor process to be used for test).

If you want to test our submission in NeurIPS2021 NetHack Challenge, please download our [checkpoint](https://arena.kakaocdn.net/brainrepo/models/brain_agent/e763eb965808ea7125796fbd903db5e3/neurips2021_nethack_kakaobrain.tar.gz) and place it at `./train_dir/nethack/`. 
When testing our submission, you don't have to set 'test.checkpoint' since it is already specified, which simplifies the evaluation command as
```bash
python eval.py cfg=configs/nethack/neurips2022/baseline.yaml actor.num_workers=$NUM_WORKERS actor.num_envs_per_worker=2 
```


## Acknowledgement

The code for overall distributed training is based on the [Sample Factory](alex-petrenko/sample-factory) repo. 
The TransformerXL-I code is based on the [Transformer-XL](https://github.com/kimiyoung/transformer-xl) repo.
The code for training baseline model is based on the [nle-sample-factory-baseline](https://github.com/Miffyli/nle-sample-factory-baseline) repo.

## Licenses
This repository is released under the MIT license, included [here](LICENSE).

This repository includes some codes from [sample-factory](https://github.com/alex-petrenko/sample-factory) 
(MIT license), [transformer-xl](https://github.com/kimiyoung/transformer-xl) (Apache 2.0 License), 
and [nle-sample-factory-baseline](https://github.com/Miffyli/nle-sample-factory-baseline) (MIT License).

## Contact

If you have any question or feedback regarding this repository, please email to contact@kakaobrain.com

## Citation

If you use Brain Agent in your research or work, please cite it as:
```
@misc{kakaobrain2022brain_agent, title = {Brain Agent}, 
author = {Donghoon Lee, Taehwan Kwon, Seungeun Rho, Daniel Wontae Nam, Jongmin Kim, Daejin Jo, and Sungwoong Kim}, 
year = {2022}, howpublished = {\url{https://github.com/kakaobrain/brain_agent}} }
```