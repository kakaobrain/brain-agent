import os
import ctypes
import multiprocessing
import random
import shutil
from pathlib import Path

from brain_agent.utils.logger import log

LEVEL_SEEDS_FILE_EXT = 'dm_lvl_seeds'


def filename_to_level(filename):
    level = filename.split('.')[0]
    level = level[1:]
    return level


def level_to_filename(level):
    filename = f'_{level}.{LEVEL_SEEDS_FILE_EXT}'
    return filename


def read_seeds_file(filename, has_keys):
    seeds = []

    with open(filename, 'r') as seed_file:
        lines = seed_file.readlines()
        for line in lines:
            try:
                if has_keys:
                    seed, cache_key = line.split(' ')
                else:
                    seed = line

                seed = int(seed)
                seeds.append(seed)
            except Exception:
                pass

    return seeds


class DmlabLevelCacheGlobal:

    def __init__(self,  cache_dir, experiment_dir, levels):
        self.cache_dir = cache_dir
        self.experiment_dir = experiment_dir

        self.all_seeds = dict()
        self.available_seeds = dict()
        self.used_seeds = dict()
        self.num_seeds_used_in_current_run = dict()
        self.locks = dict()

        for lvl in levels:
            self.all_seeds[lvl] = []
            self.available_seeds[lvl] = []
            self.num_seeds_used_in_current_run[lvl] = multiprocessing.RawValue(ctypes.c_int32, 0)
            self.locks[lvl] = multiprocessing.Lock()

        lvl_seed_files = Path(os.path.join(cache_dir, '_contributed')).rglob(f'*.{LEVEL_SEEDS_FILE_EXT}')
        for lvl_seed_file in lvl_seed_files:
            lvl_seed_file = str(lvl_seed_file)
            level = filename_to_level(os.path.relpath(lvl_seed_file, cache_dir))
            self.all_seeds[level] = read_seeds_file(lvl_seed_file, has_keys=True)
            self.all_seeds[level] = list(set(self.all_seeds[level]))

        used_lvl_seeds_dir = os.path.join(self.experiment_dir, f'dmlab_used_lvl_seeds')
        os.makedirs(used_lvl_seeds_dir, exist_ok=True)

        used_seeds_files = Path(used_lvl_seeds_dir).rglob(f'*.{LEVEL_SEEDS_FILE_EXT}')
        self.used_seeds = dict()
        for used_seeds_file in used_seeds_files:
            used_seeds_file = str(used_seeds_file)
            level = filename_to_level(os.path.relpath(used_seeds_file, used_lvl_seeds_dir))
            self.used_seeds[level] = read_seeds_file(used_seeds_file, has_keys=False)

            self.used_seeds[level] = set(self.used_seeds[level])

        for lvl in levels:
            lvl_seeds = self.all_seeds.get(lvl, [])
            lvl_used_seeds = self.used_seeds.get(lvl, [])

            lvl_remaining_seeds = set(lvl_seeds) - set(lvl_used_seeds)
            self.available_seeds[lvl] = list(lvl_remaining_seeds)

            random.shuffle(self.available_seeds[lvl])
            log.debug('Env %s has %d remaining unused seeds', lvl, len(self.available_seeds[lvl]))

    def record_used_seed(self, level, seed):
        self.num_seeds_used_in_current_run[level].value += 1

        used_lvl_seeds_dir = os.path.join(self.experiment_dir, f'dmlab_used_lvl_seeds')
        used_seeds_filename = os.path.join(used_lvl_seeds_dir, level_to_filename(level))
        os.makedirs(os.path.dirname(used_seeds_filename), exist_ok=True)

        with open(used_seeds_filename, 'a') as fobj:
            fobj.write(f'{seed}\n')

        if level not in self.used_seeds:
            self.used_seeds[level] = {seed}
        else:
            self.used_seeds[level].add(seed)

    def get_unused_seed(self, level, random_state=None):
        with self.locks[level]:
            num_used_seeds = self.num_seeds_used_in_current_run[level].value
            if num_used_seeds >= len(self.available_seeds.get(level, [])):

                while True:
                    if random_state is not None:
                        new_seed = random_state.randint(0, 2 ** 31 - 1)
                    else:
                        new_seed = random.randint(0, 2 ** 31 - 1)

                    if level not in self.used_seeds:
                        break

                    if new_seed in self.used_seeds[level]:
                        pass
                    else:
                        break
            else:
                new_seed = self.available_seeds[level][num_used_seeds]

            self.record_used_seed(level, new_seed)
            return new_seed

    def add_new_level(self, level, seed, key, pk3_path):
        with self.locks[level]:
            num_used_seeds = self.num_seeds_used_in_current_run[level].value
            if num_used_seeds < len(self.available_seeds.get(level, [])):
                log.warning('We should only add new levels to cache if we ran out of pre-generated levels (seeds)')
                log.warning(
                    'Num used seeds: %d, available seeds: %d, level: %s, seed %r, key %r',
                    num_used_seeds, len(self.available_seeds.get(level, [])), level, seed, key,
                )

            path = os.path.join(self.cache_dir, key)
            if not os.path.isfile(path):
                shutil.copyfile(pk3_path, path)

            lvl_seeds_filename = os.path.join(self.cache_dir, level_to_filename(level))
            os.makedirs(os.path.dirname(lvl_seeds_filename), exist_ok=True)
            with open(lvl_seeds_filename, 'a') as fobj:
                fobj.write(f'{seed} {key}\n')


def dmlab_ensure_global_cache_initialized(experiment_dir, levels, level_cache_dir):
    global DMLAB_GLOBAL_LEVEL_CACHE

    assert multiprocessing.current_process().name == 'MainProcess', \
        'make sure you initialize DMLab cache before child processes are forked'

    log.info('Initializing level cache...')
    DMLAB_GLOBAL_LEVEL_CACHE = DmlabLevelCacheGlobal(level_cache_dir, experiment_dir, levels)
