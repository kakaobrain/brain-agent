import os
import copy

import gym
import numpy as np
from numba import njit
import nle
import cv2
from PIL import Image, ImageFont, ImageDraw

SMALL_FONT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets/Hack-Regular.ttf'))

# Mapping of 0-15 colors used.
# Taken from bottom image here. It seems about right
# https://i.stack.imgur.com/UQVe5.png
COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080",  # - flipped these ones around
    "#C0C0C0",  # | the gray-out dull stuff
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFFFF",
]


class VectorFeaturesWrapper(gym.Wrapper):
    """Create network-friendly vector features from the stuff nethack has"""

    # Hand-chosen scaling values for each blstat entry. Aims to limit them in [0, 1] range.
    BLSTAT_NORMALIZATION_STATS = np.array([
        1.0 / 79.0,     # [0] hero col
        1.0 / 21,       # [1] hero row
        0.0,            # [2] strength pct
        1.0 / 10,       # [3] strength
        1.0 / 10,       # [4] dexterity
        1.0 / 10,       # [5] constitution
        1.0 / 10,       # [6] intelligence
        1.0 / 10,       # [7] wisdom
        1.0 / 10,       # [8] charisma
        0.0,            # [9] score
        1.0 / 10,       # [10] hitpoints
        1.0 / 10,       # [11] max hitpoints
        0.0,            # [12] depth
        1.0 / 1000,     # [13] gold
        1.0 / 10,       # [14] energy
        1.0 / 10,       # [15] max energy
        1.0 / 10,       # [16] armor class
        0.0,            # [17] monster level
        1.0 / 10,       # [18] experience level
        1.0 / 100,      # [19] experience points
        1.0 / 1000,            # [20] time
        1.0,            # [21] hunger_state
        1.0 / 10,            # [22] carrying capacity
        0.0,            # [23] dungeon number
        0.0,            # [24] level number
        0.0,            # [25] condition bits  # nle v0.7.3+ only.
    ])
    BLSTATS_NORM_MEAN = np.array([38.143184, 11.350447, 19.385929, 48.681689, 16.364037, 17.475396, 13.386432, 15.515700, 11.217532, 0,
                         140.676504, 142.441755, 0, 246.591319, 64.147795, 74.791243, -14.114734, 0, 13.747976, 21255.588020,
                         31590.399317, 0, 0, 17.505681, 0, 0])
    BLSTATS_NORM_STD = np.array([19.516993, 4.555499, 3.127983, 36.879129, 2.781533, 1.698254, 4.207449, 4.062547, 3.753857, 0,
                        87.590922, 86.641331, 0, 1077.061845, 53.786700, 56.686335, 12.114323, 0, 4.810798, 27597.589867,
                        19627.790005, 0, 0, 13.921685, 0, 0])
    BLSTATS_NORM_MIN = np.array([0, 0, 8, 8, 7, 8, 4, 7, 3, 0,
                        0, 10, 0, 0, 0, 1, -42, 0, 1, 0.0,
                        1, 0, 0, 1, 0, 0])
    BLSTATS_NORM_MAX = np.array([78, 21, 25, 118, 23, 22, 24, 25, 20, 0,
                        442, 442, 0, 64228, 347, 347, 10, 0, 30, 405504.0,
                        89228, 0, 0, 53, 0, 0])

    TTYREC_BLSTATS_INDICES = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19]

    # Make sure we do not spook the network
    BLSTAT_CLIP_RANGE = (-5, 5)

    def __init__(self, env, cfg):
        super().__init__(env)
        self.normalize_blstats = cfg.env.normalize_blstats
        self.cfg = cfg

        num_item_features = 0
        if self.normalize_blstats == 'default':
            num_items = VectorFeaturesWrapper.BLSTAT_NORMALIZATION_STATS.shape[0]
        else:
            num_items = len(VectorFeaturesWrapper.TTYREC_BLSTATS_INDICES)

        self.use_character_feature = 'vector_obs' in self.observation_space.spaces # if there is already 'vector_feature' in env.observation_space, it supposed to use character_feature
        if self.use_character_feature:
            num_items = self.observation_space['vector_obs'].shape[0]

        self.use_item_feature = 'item_feature' in self.observation_space.spaces and self.cfg.model.encoder.serialize_to_vobs # going to concat serialized item_feature to vector_obs
        if self.use_item_feature:
            num_items += np.prod(self.observation_space['item_feature'].shape)
            self.observation_space.spaces.pop('item_feature')

        self.use_spell_feature = 'spell_feature' in self.observation_space.spaces and self.cfg.model.encoder.serialize_to_vobs  # going to concat serialized item_feature to vector_obs
        if self.use_spell_feature:
            num_items += np.prod(self.observation_space['spell_feature'].shape)
            self.observation_space.spaces.pop('spell_feature')

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces['vector_obs'] = gym.spaces.Box(
            low=VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[0],
            high=VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[1],
            shape=(num_items,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _create_vector_obs(self, obs):
        if not self.use_character_feature and not ('vector_obs' in obs):
            if self.normalize_blstats == 'zscore':
                blstats = obs["blstats"][VectorFeaturesWrapper.TTYREC_BLSTATS_INDICES]
                obs_vector = (blstats - self.BLSTATS_NORM_MEAN[self.TTYREC_BLSTATS_INDICES]) / self.BLSTATS_NORM_STD[self.TTYREC_BLSTATS_INDICES]
            elif self.normalize_blstats == 'minmax':
                blstats = obs["blstats"][VectorFeaturesWrapper.TTYREC_BLSTATS_INDICES]
                obs_vector = (blstats - self.BLSTATS_NORM_MIN[self.TTYREC_BLSTATS_INDICES]) / (self.BLSTATS_NORM_MAX[self.TTYREC_BLSTATS_INDICES] - self.BLSTATS_NORM_MIN[self.TTYREC_BLSTATS_INDICES])
            elif self.normalize_blstats == 'default':
                blstats = obs["blstats"]
                norm_stats = VectorFeaturesWrapper.BLSTAT_NORMALIZATION_STATS
                obs_vector = blstats * norm_stats
            else:
                raise NotImplementedError
            np.clip(
                obs_vector,
                VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[0],
                VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[1],
                out=obs_vector
            )
        else:
            obs_vector = obs['vector_obs']

        if self.use_item_feature or 'item_feature' in obs and self.cfg.model.encoder.serialize_to_vobs:
            obs_vector = np.concatenate([obs_vector, obs['item_feature'].reshape(-1)], axis=-1)
            obs.pop('item_feature')

        if self.use_spell_feature or 'spell_feature' in obs and self.cfg.model.encoder.serialize_to_vobs:
            obs_vector = np.concatenate([obs_vector, obs['spell_feature'].reshape(-1)], axis=-1)
            obs.pop('spell_feature')

        obs["vector_obs"] = obs_vector

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._create_vector_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._create_vector_obs(obs)
        return obs

    ## for submission below here
    def on_reset_submission(self, env, obs):
        obs = self._create_vector_obs(obs)
        return obs

@njit
def _tile_characters_to_image(
        out_image,
        chars,
        colors,
        output_height_chars,
        output_width_chars,
        char_array,
        offset_h,
        offset_w
):
    """
    Build an image using cached images of characters in char_array to out_image
    """
    char_height = char_array.shape[3]
    char_width = char_array.shape[4]
    for h in range(output_height_chars):
        h_char = h + offset_h
        # Stuff outside boundaries is not visible, so
        # just leave it black
        if h_char < 0 or h_char >= chars.shape[0]:
            continue
        for w in range(output_width_chars):
            w_char = w + offset_w
            if w_char < 0 or w_char >= chars.shape[1]:
                continue
            char = chars[h_char, w_char]
            color = colors[h_char, w_char]
            h_pixel = h * char_height
            w_pixel = w * char_width
            out_image[:, h_pixel:h_pixel + char_height, w_pixel:w_pixel + char_width] = char_array[char, color]


class RenderCharImagesWithNumpyWrapper(gym.Wrapper):
    """
    Render characters as images, using PIL to render characters like we humans see on screen
    but then some caching and numpy stuff to speed up things.
    To speed things up, crop image around the player.
    """

    def __init__(self, env, font_size=9, crop_size=None, rescale_font_size=None, use_tty_chars_colors=True):
        super().__init__(env)
        self.char_array = self._initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)

        self.crop_size = crop_size
        self.use_tty_chars_colors = use_tty_chars_colors

        if crop_size is None or crop_size == 0:
            # Render full "obs"
            if self.use_tty_chars_colors:
                old_obs_space = self.env.observation_space["tty_chars"]
            else:
                old_obs_space = self.env.observation_space["chars"]
            self.output_height_chars = old_obs_space.shape[0]
            self.output_width_chars = old_obs_space.shape[1]
        else:
            # Render only crop region
            self.half_crop_size = crop_size // 2
            self.output_height_chars = crop_size
            self.output_width_chars = crop_size
        self.chw_image_shape = (
            3,
            self.output_height_chars * self.char_height,
            self.output_width_chars * self.char_width
        )

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces['obs'] = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.chw_image_shape,
            dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _initialize_char_array(self, font_size, rescale_font_size):
        """Draw all characters in PIL and cache them in numpy arrays
        if rescale_font_size is given, assume it is (width, height)
        Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
        """
        font = ImageFont.truetype(SMALL_FONT_PATH, font_size)
        dummy_text = "".join([(chr(i) if chr(i).isprintable() else " ") for i in range(256)])
        _, _, image_width, image_height = font.getbbox(dummy_text)
        # Above can not be trusted (or its siblings)....
        image_width = int(np.ceil(image_width / 256) * 256)
        if rescale_font_size:
            char_width = rescale_font_size[0]
            char_height = rescale_font_size[1]
        else:
            char_width = image_width // 256
            char_height = image_height

        char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)
        image = Image.new("RGB", (image_width, image_height))
        image_draw = ImageDraw.Draw(image)

        for color_index in range(16):
            image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))

            image_draw.text((0, 0), dummy_text, fill=COLORS[color_index], spacing=0)

            arr = np.array(image).copy()
            arrs = np.array_split(arr, 256, axis=1)
            for char_index in range(256):
                char = arrs[char_index]
                if rescale_font_size:
                    char = cv2.resize(char, rescale_font_size, interpolation=cv2.INTER_AREA)
                char_array[char_index, color_index] = char
        return char_array

    def _render_text_to_image(self, obs):
        if self.use_tty_chars_colors:
            chars = obs["tty_chars"]
            colors = obs["tty_colors"]
        else:
            chars = obs["chars"]
            colors = obs["colors"]
        offset_w = 0
        offset_h = 0
        if self.crop_size:
            # Center around player
            center_x, center_y = obs["blstats"][:2]
            offset_h = center_y - self.half_crop_size
            offset_w = center_x - self.half_crop_size

        out_image = np.zeros(self.chw_image_shape, dtype=np.uint8)

        _tile_characters_to_image(
            out_image=out_image,
            chars=chars,
            colors=colors,
            output_height_chars=self.output_height_chars,
            output_width_chars=self.output_width_chars,
            char_array=self.char_array,
            offset_h=offset_h,
            offset_w=offset_w
        )
        # arr = out_image
        # img = cv2.merge((arr[2], arr[1], arr[0]))
        # cv2.imwrite('out.png', img)
        #if self.use_tty_chars_colors:
        #    _ = obs.pop("tty_chars")
        #    _ = obs.pop("tty_colors")
        #else:
        #    _ = obs.pop("chars")
        #    _ = obs.pop("colors")
        # for key, _ in list(obs.items()):
        #     if key not in self.keys_to_be_preserved:
        #         _ = obs.pop(key)

        obs["obs"] = out_image

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._render_text_to_image(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._render_text_to_image(obs)
        return obs

    ## for submission below here
    def on_reset_submission(self, env, obs):
        obs = self._render_text_to_image(obs)
        return obs