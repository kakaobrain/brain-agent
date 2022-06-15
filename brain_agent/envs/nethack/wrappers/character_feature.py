import gym

import enum

import numpy as np

import nle
from nle import nethack as nh
from brain_agent.envs.nethack.utils.number_to_binary import number_to_binary


class CharacterFeatureWrapper(gym.Wrapper):
    """Create network-friendly vector features from the stuff nethack has"""

    # Hand-chosen scaling values for each blstat entry. Aims to limit them in [0, 1] range.
    BLSTAT_NORMALIZATION_STATS_WITHOUT_CONDITION = np.array([
        1.0 / 79.0,  # hero col
        1.0 / 21,  # hero row
        0.0,  # strength pct
        1.0 / 10,  # strength
        1.0 / 10,  # dexterity
        1.0 / 10,  # constitution
        1.0 / 10,  # intelligence
        1.0 / 10,  # wisdom
        1.0 / 10,  # charisma
        0.0,  # score
        1.0 / 10,  # hitpoints
        1.0 / 10,  # max hitpoints
        0.0,  # depth
        1.0 / 1000,  # gold
        1.0 / 10,  # energy
        1.0 / 10,  # max energy
        1.0 / 10,  # armor class
        0.0,  # monster level
        1.0 / 10,  # experience level
        1.0 / 100,  # experience points
        1.0 / 1000,  # time
        1.0,  # hunger_state
        1.0,  # carrying capacity
        0.0,  # dungeon number
        0.1,  # level number
    ])
    D_CONDITION = 30
    D_ROLE = 13
    D_RACE = 5
    D_ALIGNEMT = 3
    D_GENDER = 1

    # Make sure we do not spook the network
    BLSTAT_CLIP_RANGE = (-5, 5)

    def __init__(self, env):
        super().__init__(env)
        num_items = (
            self.BLSTAT_NORMALIZATION_STATS_WITHOUT_CONDITION.shape[0]
            + self.D_ROLE + self.D_RACE + self.D_ALIGNEMT + self.D_GENDER
            + self.D_CONDITION  # blstats(25) + condition (30)
        )

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces['vector_obs'] = gym.spaces.Box(
            low=self.BLSTAT_CLIP_RANGE[0],
            high=self.BLSTAT_CLIP_RANGE[1],
            shape=(num_items,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _get_charecter_feature_on_reset(self, env):
        obs = env.step(25)  # get check attribute
        str_attribute = obs[0]['tty_chars'].reshape(-1).tostring().decode('latin_1').lower()
        env.step(99); env.step(99)  # pass 2 pages of explanation by entering 'space' action

        role, race, alignment, gender = None, None, None, None
        obs_role = np.zeros(self.D_ROLE)
        obs_race = np.zeros(self.D_RACE)
        obs_alignment = np.zeros(self.D_ALIGNEMT)
        obs_gender = np.zeros(self.D_GENDER)

        # Check role
        roles = [ 'archeologist', 'barbarian', 'cave', 'healer', 'knight', 'monk', 'priest', 'ranger', 'rogue', 'samurai', 'tourist', 'valkyrie', 'wizard' ]
        self.roles = roles
        for _role in roles:
            if _role in str_attribute:
                role = _role
        obs_role[roles.index(role)] = 1.0

        # Check race
        races = ['human', 'elven', 'gnomish', 'orcish', 'dwarven']
        for _race in races:
            if _race in str_attribute:
                race = _race
        obs_race[races.index(race)] = 1.0

        # Check moral alignment
        alignments = [' lawful', ' neutral', ' chaotic'] # front space should be included since all words appear on str_attribute regardless of alignment
        for _alignment in alignments:
            if _alignment in str_attribute:
                alignment = _alignment
        obs_alignment[alignments.index(alignment)] = 1.0

        # Check gender
        genders = ['male', 'female']
        str_females = ['female', 'woman', 'ess', 'valkyrie']
        str_males = ['male', 'man', 'priest']
        for _str_male in str_males:
            if _str_male in str_attribute:
                gender = 'male'
        for _str_female in str_females:
            if _str_female in str_attribute:
                gender = 'female'
        if gender == 'male':
            obs_gender[0] = 1.0

        self.role = role
        assert not (
                role is None
                or race is None
                or alignment is None
                or gender is None
        ), f"role: {role}, race: {race}, alignment: {alignment}, gender: {gender} from {str_attribute}"
        obs_chracter = np.concatenate([obs_role, obs_race, obs_alignment, obs_gender], axis=0)
        return obs_chracter

    def _create_character_feature(self, obs):
        obs_vector = obs["blstats"][:-1] * self.BLSTAT_NORMALIZATION_STATS_WITHOUT_CONDITION
        np.clip(
            obs_vector,
            self.BLSTAT_CLIP_RANGE[0],
            self.BLSTAT_CLIP_RANGE[1],
            out=obs_vector
        )
        condition_array = number_to_binary(obs["blstats"][-1], n_bit = self.D_CONDITION)
        obs_vector = np.concatenate(
            [
                obs_vector,
                self.character_feature,
                condition_array
            ],
            axis = 0
        )

        obs["vector_obs"] = obs_vector

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._create_character_feature(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.character_feature = self._get_charecter_feature_on_reset(self.env)
        obs = self._create_character_feature(obs)

        return obs

    ## for submission below here
    def on_reset_submission(self, env, obs, is_holding_aicrowd=False):
        self.is_holding_aicrowd = is_holding_aicrowd
        self.env_aicrowd = env
        #self.step = lambda action: self.step_submission(action, self.env_aicrowd)

        self.character_feature = self._get_charecter_feature_on_reset(env)
        obs = self._create_character_feature(obs)

        return obs