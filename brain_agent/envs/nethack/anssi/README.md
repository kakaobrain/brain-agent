
## NLE challenge baseline using sample-factory
  * https://github.com/Miffyli/nle-sample-factory-baseline/blob/main/models.py
  * 실행 전 `pip install numba` 패키지 설치가 필요합니다


### Command
  * https://github.com/Miffyli/nle-sample-factory-baseline/blob/main/train_dir/baseline-code/cfg.json
    * 여기 설정과 최대한 비슷하게 아래 커맨드를 생성하였습니다. (다른게 있는지 더 확인 필요)
    * 원코드에서는 `--rnn_type=gru` 를 사용했지만 여기서는 에러가 발생하여 일단 `--rnn_type=lstm` 으로 사용 중
    * `--encoder_custom=nle_obs_vector_encoder` 를 지정하여 모델 사용 가능
    * `--train_dir={train_dir}`, `--experiment={experiment_name}` 를 적절히 변경 후 실행
    ```bash
    python -m dist.launch --nnodes=1 --nproc_per_node=4 --node_rank=0 -m brain_agent.algorithms.appo.train_appo \
        --env=nethack_challenge --train_for_env_steps=1000000000 --algo=APPO --optimizer_type=adam --rmsprop_eps=0.000001 \
        --learning_rate=0.0001 --exploration_loss=entropy --exploration_loss_coeff=0.003 --use_rnn=True \
        --rollout=32 --recurrence=32 --max_grad_norm=4.0 --encoder_extra_fc_layers=1 --with_pbt=False \
        --set_workers_cpu_affinity=True --max_policy_lag=10000 --loss_type=sum_ori --use_popart=False --train_dir={train_dir} \
        --num_workers=24 --num_envs_per_worker=20 --batch_size=4096 --num_batches_per_iteration=1 --num_policies=1 --rnn_type=lstm --rnn_num_layers=1 \
        --normalize_reward=True --use_ppo=False --packed_seq=False --scheduler=lineardecay --experiment={experiment_name} --encoder_custom=nle_obs_vector_encoder --save_milestones_step=100000000 \
        --stats_avg=1000 --reward_scale=0.1 --reward_clip=10.0 --obs_scale=255.0
    ```
