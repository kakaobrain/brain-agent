# Brain Agent
***Brain Agent*** is a distributed agent learning system for large-scale and multi-task reinforcement learning, developed by [Kakao Brain](https://www.kakaobrain.com/). 
Brain Agent is based on the V-trace actor-critic framework [IMPALA](https://arxiv.org/abs/1802.01561) and modifies [Sample Factory](https://github.com/alex-petrenko/sample-factory) to learn a policy network using multiple GPUs and CPUs with a high throughput rate during training.

Especially, Brain Agent supports the [PopArt](https://arxiv.org/abs/1809.04474) normalization as well as the [TransformerXL-I](https://proceedings.mlr.press/v119/parisotto20a.html) core of the policy network. In addition, for fast and stable training, it allows to make use of the reconstruction loss by applying the auxiliary decoder to the encoder of the policy network.

With these advanced techniques Brain Agent obtains the state-of-the-art result on [DeepMindLab](https://github.com/deepmind/lab) benchmark (DMLab-30 multi-task training) in terms of human normalized score after training for 20 billion frames.
In specific, our transformer-based policy network has 28 million parameters and was trained on 16 V100 GPUs for 7 days.

We release the source code and trained agent to facilitate future research on DMLab benchmark.

The agent successfully obtained near SOTA score of **91.25** (testing) in terms of capped human normalized score on DMLab-30 
(multi-task mode), after training for 20B frames.


## Installation
This code has been developed under Python 3.7, Pytorch 1.9.0 and CUDA 11.1.
For DMLab environment, please follow installation instructions from [DMLab Github](https://github.com/deepmind/lab/blob/master/docs/users/build.md).

Please run the following command to install the other necessary dependencies.
```bash
pip install -r requirements.txt
```


## Training
The script `train.py` is used for training. Example usage to train a model on DMLab-30 with our 28M 
transformer-based policy network on 4 nodes having 4 GPUs on each node is
```bash
sleep 120; python -m dist_launch --nnodes=4 --node_rank=0 --nproc_per_node=4 --master_addr=$MASTER_ADDR -m train \ 
  cfg=configs/trxl_recon_train.yaml train_dir=$TRAIN_DIR experiment=EXPERIMENT_DIR 
sleep 120; python -m dist_launch --nnodes=4 --node_rank=1 --nproc_per_node=4 --master_addr=$MASTER_ADDR -m train \ 
  cfg=configs/trxl_recon_train.yaml train_dir=$TRAIN_DIR experiment=EXPERIMENT_DIR
sleep 120; python -m dist_launch --nnodes=4 --node_rank=2 --nproc_per_node=4 --master_addr=$MASTER_ADDR -m train \ 
  cfg=configs/trxl_recon_train.yaml train_dir=$TRAIN_DIR experiment=EXPERIMENT_DIR
sleep 120; python -m dist_launch --nnodes=4 --node_rank=3 --nproc_per_node=4 --master_addr=$MASTER_ADDR -m train \ 
  cfg=configs/trxl_recon_train.yaml train_dir=$TRAIN_DIR experiment=EXPERIMENT_DIR
```

## Evaluation
### Trained Agent on DMLab-30
We provide the checkpoint of our trained agent on DMLab-30. The overall performances of our agent and the previous
state-of-the-art models are as follows.

|  Model | Mean HNS  | Median HNS | Mean Capped HNS |
|:----------:|:--------:|:----------:|:---------------:|
| [MERLIN](https://arxiv.org/pdf/1803.10760.pdf) | 115.2  | - | 89.4 |
| [GTrXL](https://proceedings.mlr.press/v119/parisotto20a.html) | 117.6 | - | 89.1 |
| [CoBERL](https://arxiv.org/pdf/2107.05431.pdf) | 115.47  | 110.86 |- |
| [R2D2+](https://openreview.net/pdf?id=r1lyTjAqYX) | -  | 99.5 | 85.7 |
| [LASER](https://arxiv.org/abs/1909.11583) | -  | 97.2 | 81.7 |
| [PBL](https://arxiv.org/pdf/2004.14646.pdf) | 104.16  | - | 81.5 |
| [PopArt-IMPALA](https://arxiv.org/abs/1809.04474) | -  | - | 72.8 |
| [IMPALA](https://arxiv.org/abs/1802.01561) | -  | - | 58.4 |
| Ours ([20B checkpoint](https://arena.kakaocdn.net/brainrepo/models/brain_agent/84ac1c594b8eb95e7fd9879d6172f99b/trxl_recon_20b.pth)) | 123.60 ± 0.84 | 108.63	± 1.20 | **91.25 ± 0.41**|

Our results were obtained by 3 runs with different random seeds where each run carried out 100 episodes for each task (level).

The detailed break down HNS scores by our agent are as follows:

| Level | HNS |
|:-----:|:---:|
rooms_collect_good_objects_(train / test)		|	97.58	±   0.20 / 89.39	±	1.42		|
rooms_exploit_deferred_effects_(train / test)		|	38.86	±   3.48 / 4.04	±	0.89		|
rooms_select_nonmatching_object		|	99.52	±	0.97		|
rooms_watermaze		|	111.20	±	2.29		|
rooms_keys_doors_puzzle		|	61.24	±	9.09		|
language_select_described_object		|	155.35	±	0.17		|
language_select_located_object		|	252.04	±	0.31		|
language_execute_random_task		|	145.21	±	0.36		|
language_answer_quantitative_question		|	163.72	±	1.36		|
lasertag_one_opponent_small		|	249.99	±	6.64		|
lasertag_three_opponents_small		|	246.68	±	5.99		|
lasertag_one_opponent_large		|	82.55	±	2.15		|
lasertag_three_opponents_large		|	96.54	±	0.67		|
natlab_fixed_large_map		|	120.53	±	1.79		|
natlab_varying_map_regrowth		|	108.14	±	1.25		|
natlab_varying_map_randomized		|	85.53	±	6.69		|
skymaze_irreversible_path_hard		|	61.63	±	2.52		|
skymaze_irreversible_path_varied		|	81.31	±	2.34		|
psychlab_arbitrary_visuomotor_mapping		|	101.82	±	0.19		|
psychlab_continuous_recognition		|	102.46	±	0.32		|
psychlab_sequential_comparison		|	75.74	±	0.58		|
psychlab_visual_search		|	101.91	±	0.00		|
explore_object_locations_small		|	123.54	±	2.61		|
explore_object_locations_large		|	115.43	±	1.64		|
explore_obstructed_goals_small		|	166.75	±	3.63		|
explore_obstructed_goals_large		|	153.44	±	3.20		|
explore_goal_locations_small		|	177.16	±	0.37		|
explore_goal_locations_large		|	160.39	±	3.32		|
explore_object_rewards_few		|	109.58	±	3.53		|
explore_object_rewards_many		|	105.15	±	0.75		|


- Learning curves of our 20B checkpoint on dmlab-30
<div align="center">
 <img width="800" alt="Learning Curve" src="assets/learning_curve.png"/>
</div>


### Evaluation Command
The script `eval.py` is used for evaluation. Example usage to test the provided checkpoint, which is located in $CHECKPOINT_FILE_PATH, is

```bash
python eval.py cfg=configs/trxl_recon_eval.yaml train_dir=$TRAIN_DIR experiment=$EXPERIMENT_DIR model.checkpoint=$CHECKPOINT_FILE_PATH 
```

## Acknowledgement

The code for overall distributed training is based on the [Sample Factory](alex-petrenko/sample-factory) repo, and the TransformerXL-I code is based on the [Transformer-XL](https://github.com/kimiyoung/transformer-xl) repo.

## Licenses
This repository is released under the MIT license, included [here](LICENSE).

This repository includes some codes from [sample-factory](https://github.com/alex-petrenko/sample-factory) 
(MIT license) and [transformer-xl](https://github.com/kimiyoung/transformer-xl) (Apache 2.0 License).

## Contact

If you have any question or feedback regarding this repository, please email to contact@kakaobrain.com

## Citation

If you use Brain Agent in your research or work, please cite it as:
```
@misc{kakaobrain2022brain_agent, title = {Brain Agent}, 
author = {Donghoon Lee, Taehwan Kwon, Seungeun Rho, Daniel Wontae Nam, Jongmin Kim, Daejin Jo, and Sungwoong Kim}, 
year = {2022}, howpublished = {\url{https://github.com/kakaobrain/brain_agent}} }
```
