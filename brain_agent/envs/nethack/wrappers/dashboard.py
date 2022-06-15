import glob
import os
import csv
import bz2
import zipfile
import copy
from collections import OrderedDict

import gym
import nle.env
from nle import nethack
from brain_agent.utils.utils import log


class NetHackDashboardWrapper(gym.Wrapper):
    """
    * Dashboard 서버 실행하기
    https://github.com/facebookresearch/nle/blob/master/nle/dashboard/README.md

      $ cd sample_factory/sample_factory/envs/nethack/dashboard
      $ sed -i "s/3000/$PORT1/g" config.js && echo "PORT: $PORT1"  # serverPort를 사용가능한 외부 포트로 변경
      $ apt-get install npm  # npm 패키지 설치
      $ npm install # 의존 패키지 설치하는 과정으로 한번만 실행
      $ npm start   # 서버 시작
      $ echo "http://$HOSTNAME:$PORT1"  # HTTP 접속 가능한 주소 출력

      * Dashboard 사용 방법
        - `Data Path`에 `stats.csv`, `stats.zip`가 포함된 경로를 입력하고 "Load" 클릭

    """

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.stats_csv_path = os.path.join(self.env.savedir, 'stats.csv')
        self.env = env
        self._last_obs = None

        nethack_env = self.env
        while not isinstance(nethack_env, nethack.Nethack) and hasattr(nethack_env, 'env'):
            nethack_env = nethack_env.env
        assert isinstance(nethack_env, nethack.Nethack)

        nle_env = self.env
        while not isinstance(nle_env, nle.env.NLE) and hasattr(nle_env, 'env'):
            nle_env = nle_env.env
        assert isinstance(nle_env, nle.env.NLE)

        self._nethack_env = nethack_env
        self._nle_env = nle_env
        assert hasattr(self._nethack_env, '_ttyrec')

    def _collect_stats(self, observation, end_status):
        blstats = observation['blstats']
        ttyrec_path = self._nethack_env._ttyrec

        # https://github.com/facebookresearch/nle/blob/ceee6396797c5fe00eac66aa909af48e5ee8b04d/src/eat.c#L75-L76
        hunger_state = ["Satiated", "", "Hungry", "Weak", "Fainting", "Fainted ", "Starved "]

        # blstats indices: https://github.com/facebookresearch/nle/blob/v0.7.3/include/nleobs.h#L16-L42
        stats = OrderedDict(
            episode=self._nle_env._episode,
            end_status=end_status,
            score=blstats[nethack.NLE_BL_SCORE],
            time=blstats[nethack.NLE_BL_TIME],  # https://nethackwiki.com/wiki/Turn
            steps=self._nle_env._steps,  # number of step() calls
            hp=blstats[nethack.NLE_BL_HP],
            exp=blstats[nethack.NLE_BL_EXP],
            exp_lev=blstats[nethack.NLE_BL_XP],
            gold=blstats[nethack.NLE_BL_GOLD],
            hunger=hunger_state[blstats[nethack.NLE_BL_HUNGER]],
            ttyrec=ttyrec_path.split('/')[-1].replace('.bz2', '') if isinstance(ttyrec_path, str) else '',
        )

        if ttyrec_path and os.path.exists(ttyrec_path):
            add_header = not os.path.exists(self.stats_csv_path)
            with open(self.stats_csv_path, 'a', buffering=1) as writer:
                csv_writer = csv.DictWriter(writer, fieldnames=stats.keys())
                if add_header:
                    csv_writer.writeheader()
                csv_writer.writerow(stats)
        return stats

    def step(self, action: int):
        observation, reward, done, info = self.env.step(action)

        if done:
            # done 인 경우 observation['blstats']은 빈 값이 되기 때문에 마지막 observation을 사용
            stats = self._collect_stats(self._last_obs, info['end_status'])
        if observation['blstats'][:2].sum() > 0:
            self._last_obs = copy.deepcopy(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        ttyrec_path = self._nethack_env._ttyrec  # Must get the ttyrec_path before reset()
        observations = self.env.reset(**kwargs)

        # Nethack env에서는 ttyrec 파일을 bz2로 압축하고 Dashboard에서는 .ttyrec 파일만 재생할 수 있어서 다시 압축 해제 필요
        # env.reset()이 호출되어야 ttyrec 파일 쓰기가 완료되므로 env.reset() 이후에 bz2 압축 해제
        if ttyrec_path and ttyrec_path.endswith('.bz2') and os.path.exists(ttyrec_path):
            if os.path.getsize(ttyrec_path) == 0:
                log.warning(f'{ttyrec_path} is empty. it will be ignored.')
            else:
                with open(ttyrec_path.replace('.bz2', ''), 'wb') as new_file, bz2.BZ2File(ttyrec_path, 'rb') as file:
                    for data in iter(lambda: file.read(100 * 1024), b''):
                        new_file.write(data)

                with zipfile.ZipFile(os.path.join(os.path.dirname(ttyrec_path), 'stats.zip'), 'w') as zf:
                    for file in glob.glob(os.path.join(os.path.dirname(ttyrec_path), '*.ttyrec')):
                        zf.write(file,
                                 arcname=os.path.basename(file),
                                 compress_type=zipfile.ZIP_DEFLATED)
        return observations
