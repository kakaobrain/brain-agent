import torch

from torch import nn
from brain_agent.core.models.model_abc import  EncoderBase

from brain_agent.utils.utils import log
import collections
from torch.nn import functional as F
from einops import rearrange
from nle.nethack import *
import numpy as np

NUM_GLYPHS = MAX_GLYPH
BLSTATS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 20, 21, 24]
NUM_FEATURES = len(BLSTATS_INDICES)
PAD_CHAR = 0
NUM_CHARS = 256


def get_action_space_mask(action_space, reduced_action_space):
    mask = np.array([int(a in reduced_action_space) for a in action_space])
    return torch.Tensor(mask)


def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)


def select(embedding_layer, x, use_index_select):
    """Use index select instead of default forward to possible speed up embedding."""
    if use_index_select:
        out = embedding_layer.weight.index_select(0, x.view(-1))
        # handle reshaping x to 1-d and output back to N-d
        return out.view(x.shape + (-1,))
    else:
        return embedding_layer(x)


class GlyphGroup(enum.IntEnum):
    # See display.h in NetHack.
    MON = 0
    PET = 1
    INVIS = 2
    DETECT = 3
    BODY = 4
    RIDDEN = 5
    OBJ = 6
    CMAP = 7
    EXPLODE = 8
    ZAP = 9
    SWALLOW = 10
    WARNING = 11
    STATUE = 12


def id_pairs_table():
    """Returns a lookup table for glyph -> NLE id pairs."""
    table = np.zeros([MAX_GLYPH, 2], dtype=np.int32)

    num_nle_ids = 0

    for glyph in range(GLYPH_MON_OFF, GLYPH_PET_OFF):
        table[glyph] = (glyph, GlyphGroup.MON)
        num_nle_ids += 1

    for glyph in range(GLYPH_PET_OFF, GLYPH_INVIS_OFF):
        table[glyph] = (glyph - GLYPH_PET_OFF, GlyphGroup.PET)

    for glyph in range(GLYPH_INVIS_OFF, GLYPH_DETECT_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.INVIS)
        num_nle_ids += 1

    for glyph in range(GLYPH_DETECT_OFF, GLYPH_BODY_OFF):
        table[glyph] = (glyph - GLYPH_DETECT_OFF, GlyphGroup.DETECT)

    for glyph in range(GLYPH_BODY_OFF, GLYPH_RIDDEN_OFF):
        table[glyph] = (glyph - GLYPH_BODY_OFF, GlyphGroup.BODY)

    for glyph in range(GLYPH_RIDDEN_OFF, GLYPH_OBJ_OFF):
        table[glyph] = (glyph - GLYPH_RIDDEN_OFF, GlyphGroup.RIDDEN)

    for glyph in range(GLYPH_OBJ_OFF, GLYPH_CMAP_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.OBJ)
        num_nle_ids += 1

    for glyph in range(GLYPH_CMAP_OFF, GLYPH_EXPLODE_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.CMAP)
        num_nle_ids += 1

    for glyph in range(GLYPH_EXPLODE_OFF, GLYPH_ZAP_OFF):
        id_ = num_nle_ids + (glyph - GLYPH_EXPLODE_OFF) // MAXEXPCHARS
        table[glyph] = (id_, GlyphGroup.EXPLODE)

    num_nle_ids += EXPL_MAX

    for glyph in range(GLYPH_ZAP_OFF, GLYPH_SWALLOW_OFF):
        id_ = num_nle_ids + (glyph - GLYPH_ZAP_OFF) // 4
        table[glyph] = (id_, GlyphGroup.ZAP)

    num_nle_ids += NUM_ZAP

    for glyph in range(GLYPH_SWALLOW_OFF, GLYPH_WARNING_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.SWALLOW)
    num_nle_ids += 1

    for glyph in range(GLYPH_WARNING_OFF, GLYPH_STATUE_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.WARNING)
        num_nle_ids += 1

    for glyph in range(GLYPH_STATUE_OFF, MAX_GLYPH):
        table[glyph] = (glyph - GLYPH_STATUE_OFF, GlyphGroup.STATUE)

    return table


class NethackReplayEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        self.flags = cfg

        self.observation_shape = obs_space['chars'].shape

        self.H = self.observation_shape[0]
        self.W = self.observation_shape[1]

        self.h_dim = self.flags.hidden_size

        # GLYPH + CROP MODEL
        self.glyph_model = GlyphEncoder(self.flags, self.H, self.W, self.flags.crop_dim)

        # MESSAGING MODEL
        self.msg_model = MessageEncoder(
            self.flags.msg_hidden_dim, self.flags.msg_embedding_dim
        )

        # BLSTATS MODEL
        self.blstats_model = BLStatsEncoder(NUM_FEATURES, self.flags.embedding_dim)

        out_dim = (
                self.blstats_model.hidden_dim
                + self.glyph_model.hidden_dim
                + self.msg_model.hidden_dim
        )

        self.init_fc_blocks(out_dim)

        # self.fc = nn.Sequential(
        #     nn.Linear(out_dim, self.h_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.h_dim, self.h_dim),
        #     nn.ReLU(),
        # )

    def device_and_type_for_input_tensor(self, input_tensor_name):
        if input_tensor_name == "glyphs":
            dtype = torch.int16
        elif input_tensor_name == "chars":
            dtype = torch.uint8
        elif input_tensor_name == "colors":
            dtype = torch.uint8
        elif input_tensor_name == "specials":
            dtype = torch.uint8
        elif input_tensor_name == "blstats":
            dtype = torch.int64
        elif input_tensor_name == "message":
            dtype = torch.uint8
        elif input_tensor_name == "tty_chars":
            dtype = torch.uint8
        elif input_tensor_name == "tty_colors":
            dtype = torch.int8
        elif input_tensor_name == "tty_cursor":
            dtype = torch.uint8
        elif input_tensor_name == "inv_glyphs":
            dtype = torch.int16
        elif input_tensor_name == "inv_strs":
            dtype = torch.uint8
        elif input_tensor_name == "inv_letters":
            dtype = torch.uint8
        elif input_tensor_name == "inv_oclasses":
            dtype = torch.uint8
        else:
            dtype = torch.float32
        return self.model_device(), dtype

    def forward(self, inputs, learning=False):
        if len(inputs["glyphs"].shape) == 3:  # (BT) x H x W
            for (k, v) in inputs.items():
                inputs[k] = v.unsqueeze(1)

        T, B, H, W = inputs["glyphs"].shape

        reps = []

        # -- [B' x K] ; B' == (T x B)
        glyphs_rep = self.glyph_model(inputs)
        assert not glyphs_rep.isnan().any(), glyphs_rep
        reps.append(glyphs_rep)  # B x 13920

        # -- [B' x K]
        char_rep = self.msg_model(inputs)
        reps.append(char_rep)  # B x 64

        # -- [B' x K]
        features_emb = self.blstats_model(inputs)
        reps.append(features_emb)  # B x 64

        # -- [B' x K]
        st = torch.cat(reps, dim=1)  # B x 14048

        # -- [B' x K]
        # st = self.fc(st) # B x 256

        st = self.forward_fc_blocks(st)

        return st


class GlyphEncoder(nn.Module):
    """This glyph encoder first breaks the glyphs (integers up to 6000) to a
    more structured representation based on the qualities of the glyph: chars,
    colors, specials, groups and subgroup ids..
       Eg: invisible hell-hound: char (d), color (red), specials (invisible),
                                 group (monster) subgroup id (type of monster)
       Eg: lit dungeon floor: char (.), color (white), specials (none),
                              group (dungeon) subgroup id (type of dungeon)

    An embedding is provided for each of these, and the embeddings are
    concatenated, before encoding with a number of CNN layers.  This operation
    is repeated with a crop of the structured reprentations taken around the
    characters position, and the two representations are concatenated
    before returning.
    """

    def __init__(self, flags, rows, cols, crop_dim, device=None):
        super(GlyphEncoder, self).__init__()

        self.crop = Crop(rows, cols, crop_dim, crop_dim, device)
        K = flags.embedding_dim  # number of input filters
        L = flags.layers  # number of convnet layers

        assert (
                K % 8 == 0
        ), "This glyph embedding format needs embedding dim to be multiple of 8"
        unit = K // 4
        self.chars_embedding = nn.Embedding(256, 2 * unit)
        self.colors_embedding = nn.Embedding(16, unit)
        self.specials_embedding = nn.Embedding(256, unit)

        self.id_pairs_table = nn.parameter.Parameter(
            torch.from_numpy(id_pairs_table()), requires_grad=False
        )
        num_groups = self.id_pairs_table.select(1, 1).max().item() + 1
        num_ids = self.id_pairs_table.select(1, 0).max().item() + 1

        # self.groups_embedding = nn.Embedding(num_groups, unit)
        # self.ids_embedding = nn.Embedding(num_ids, 3 * unit)

        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        self.output_filters = 8

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [self.output_filters]

        h, w, c = rows, cols, crop_dim
        conv_extract, conv_extract_crop = [], []
        for i in range(L):
            conv_extract.append(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
            )
            conv_extract.append(nn.ELU())

            conv_extract_crop.append(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
            )
            conv_extract_crop.append(nn.ELU())

            # Keep track of output shapes
            h = conv_outdim(h, F, P, S)
            w = conv_outdim(w, F, P, S)
            c = conv_outdim(c, F, P, S)

        self.hidden_dim = (h * w + c * c) * self.output_filters
        self.extract_representation = nn.Sequential(*conv_extract)
        self.extract_crop_representation = nn.Sequential(*conv_extract_crop)
        self.select = lambda emb, x: select(emb, x, flags.use_index_select)

    def glyphs_to_ids_groups(self, glyphs):
        T, B, H, W = glyphs.shape
        ids_groups = self.id_pairs_table.index_select(0, glyphs.view(-1).long())
        ids = ids_groups.select(1, 0).view(T, B, H, W).long()
        groups = ids_groups.select(1, 1).view(T, B, H, W).long()
        return [ids, groups]

    def forward(self, inputs):
        T, B, H, W = inputs["glyphs"].shape
        ids, groups = self.glyphs_to_ids_groups(inputs["glyphs"])

        glyph_tensors = [
            self.select(self.chars_embedding, inputs["chars"].long()),
            self.select(self.colors_embedding, inputs["colors"].long()),
            self.select(self.specials_embedding, inputs["specials"].long()),
            # self.select(self.groups_embedding, groups),
            # self.select(self.ids_embedding, ids),
        ]

        glyphs_emb = torch.cat(glyph_tensors, dim=-1)
        glyphs_emb = rearrange(glyphs_emb, "T B H W K -> (T B) K H W")

        coordinates = inputs["blstats"].view(T * B, -1).float()[:, :2]
        crop_emb = self.crop(glyphs_emb, coordinates)

        glyphs_rep = self.extract_representation(glyphs_emb)
        glyphs_rep = rearrange(glyphs_rep, "B C H W -> B (C H W)")
        assert glyphs_rep.shape[0] == T * B

        crop_rep = self.extract_crop_representation(crop_emb)
        crop_rep = rearrange(crop_rep, "B C H W -> B (C H W)")
        assert crop_rep.shape[0] == T * B

        st = torch.cat([glyphs_rep, crop_rep], dim=1)
        return st


class MessageEncoder(nn.Module):
    """This model encodes the the topline message into a fixed size representation.

    It works by using a learnt embedding for each character before passing the
    embeddings through 6 CNN layers.

    Inspired by Zhang et al, 2016
    Character-level Convolutional Networks for Text Classification
    https://arxiv.org/abs/1509.01626
    """

    def __init__(self, hidden_dim, embedding_dim, device=None):
        super(MessageEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.msg_edim = embedding_dim

        self.char_lt = nn.Embedding(NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR)
        self.conv1 = nn.Conv1d(self.msg_edim, self.hidden_dim, kernel_size=7)
        self.conv2_6_fc = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # conv2
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # conv3
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3),
            nn.ReLU(),
            # conv4
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3),
            nn.ReLU(),
            # conv5
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3),
            nn.ReLU(),
            # conv6
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # fc receives -- [ B x h_dim x 5 ]
            Flatten(),
            nn.Linear(5 * self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )  # final output -- [ B x h_dim x 5 ]

    def forward(self, inputs):
        T, B, *_ = inputs["message"].shape
        messages = inputs["message"].long().view(T * B, -1)
        # [ T * B x E x 256 ]
        char_emb = self.char_lt(messages).transpose(1, 2)
        char_rep = self.conv2_6_fc(self.conv1(char_emb))
        return char_rep


class BLStatsEncoder(nn.Module):
    """This model encodes the bottom line stats into a fixed size representation.

    It works by simply using two fully-connected layers with ReLU activations.
    """

    def __init__(self, num_features, hidden_dim):
        super(BLStatsEncoder, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embed_features = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, inputs):
        T, B, *_ = inputs["blstats"].shape

        # features = inputs["blstats"][:, :, :NUM_FEATURES]
        features = inputs["blstats"][:, :, BLSTATS_INDICES]
        # -- [B' x F]
        features = features.view(T * B, -1).float()
        # -- [B x K]
        features_emb = self.embed_features(features)

        assert features_emb.shape[0] == T * B
        return features_emb


class Crop(nn.Module):
    def __init__(self, height, width, height_target, width_target, device=None):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target

        width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)
        self.register_buffer('width_grid', width_grid[None, :].expand(self.height_target, -1))

        height_grid = self._step_to_range(2 / (self.height - 1), height_target)
        self.register_buffer('height_grid', height_grid[:, None].expand(-1, self.width_target))

        if device is not None:
            self.width_grid = self.width_grid.to(device)
            self.height_grid = self.height_grid.to(device)

    def _step_to_range(self, step, num_steps):
        return torch.tensor([step * (i - num_steps // 2) for i in range(num_steps)])

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.

        Args:
           inputs [B x H x W] or [B x C x H x W]
           coordinates [B x 2] x,y coordinates

        Returns:
           [B x C x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1).float()

        assert inputs.shape[2] == self.height, "expected %d but found %d" % (
            self.height,
            inputs.shape[2],
        )
        assert inputs.shape[3] == self.width, "expected %d but found %d" % (
            self.width,
            inputs.shape[3],
        )

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        crop = torch.round(F.grid_sample(inputs, grid, align_corners=True)).squeeze(1)
        return crop


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
