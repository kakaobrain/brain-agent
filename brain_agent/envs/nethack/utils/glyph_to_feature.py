import enum
import nle
from nle import nethack as nh

import numpy as np

from brain_agent.envs.nethack.ids.action_ids import ActionIds
from brain_agent.envs.nethack.ids.misc_ids import MISCIds
from brain_agent.envs.nethack.utils.number_to_binary import number_to_binary


# TODO: import this from NLE again
NUM_OBJECTS = 453
MAXEXPCHARS = 9


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


def glyph_to_feature_table():
    DIM_GLYPH_GROUP = 13
    DIM_OBJECT_CLASS = 18
    DIM_SPECIFIC = 10
    table = np.zeros([nh.MAX_GLYPH + 1, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS + DIM_SPECIFIC], dtype = np.int8)

    num_nle_ids = 0

    # (0, 'GLYPH_MON_OFF'), 381 elements of monsters
    for glyph in range(nh.GLYPH_MON_OFF, nh.GLYPH_PET_OFF):
        table[glyph, GlyphGroup.MON] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS: ] = number_to_binary(glyph, n_bit = DIM_SPECIFIC)
        num_nle_ids += 1

    # (381, 'GLYPH_PET_OFF'), 381 elements of monsters (same as GLYPH_MON)
    for glyph in range(nh.GLYPH_PET_OFF, nh.GLYPH_INVIS_OFF):
        table[glyph, GlyphGroup.PET] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(glyph - nh.GLYPH_PET_OFF, n_bit = DIM_SPECIFIC)

    # (762, 'GLYPH_INVIS_OFF'), 1 element
    for glyph in range(nh.GLYPH_INVIS_OFF, nh.GLYPH_DETECT_OFF):
        table[glyph, GlyphGroup.INVIS] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(num_nle_ids, n_bit = DIM_SPECIFIC)
        num_nle_ids += 1

    # (763, 'GLYPH_DETECT_OFF'), 381 elements of monsters
    for glyph in range(nh.GLYPH_DETECT_OFF, nh.GLYPH_BODY_OFF):
        table[glyph, GlyphGroup.DETECT] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(glyph - nh.GLYPH_DETECT_OFF, n_bit = DIM_SPECIFIC)

    # (1144, 'GLYPH_BODY_OFF'), 381 elements of monsters
    for glyph in range(nh.GLYPH_BODY_OFF, nh.GLYPH_RIDDEN_OFF):
        table[glyph, GlyphGroup.BODY] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(glyph - nh.GLYPH_BODY_OFF, n_bit = DIM_SPECIFIC)

    # (1525, 'GLYPH_RIDDEN_OFF'), 381 elements of monsters
    for glyph in range(nh.GLYPH_RIDDEN_OFF, nh.GLYPH_OBJ_OFF):
        table[glyph, GlyphGroup.RIDDEN] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(glyph - nh.GLYPH_RIDDEN_OFF, n_bit = DIM_SPECIFIC)

    # (1906, 'GLYPH_OBJ_OFF'), 453 elements in objects. It starts counting from 382 (n_monster + n_invis)
    for glyph in range(nh.GLYPH_OBJ_OFF, nh.GLYPH_CMAP_OFF):
        table[glyph, GlyphGroup.OBJ] = 1
        idx_obj = nh.glyph_to_obj(glyph)
        class_obj = ord(nh.objclass(idx_obj).oc_class)
        table[glyph, DIM_GLYPH_GROUP + class_obj] = 1 # set object class by onehot
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(num_nle_ids, n_bit = DIM_SPECIFIC)
        num_nle_ids += 1

    # (2359, 'GLYPH_CMAP_OFF'),
    for glyph in range(nh.GLYPH_CMAP_OFF, nh.GLYPH_EXPLODE_OFF):
        table[glyph, GlyphGroup.CMAP] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(num_nle_ids, n_bit = DIM_SPECIFIC)
        num_nle_ids += 1

    # (2446, 'GLYPH_EXPLODE_OFF'),
    for glyph in range(nh.GLYPH_EXPLODE_OFF, nh.GLYPH_ZAP_OFF):
        table[glyph, GlyphGroup.EXPLODE] = 1
        id_ = num_nle_ids + (glyph - nh.GLYPH_EXPLODE_OFF) // MAXEXPCHARS
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(id_, n_bit = DIM_SPECIFIC)

    num_nle_ids += nh.EXPL_MAX

    # (2509, 'GLYPH_ZAP_OFF')
    for glyph in range(nh.GLYPH_ZAP_OFF, nh.GLYPH_SWALLOW_OFF):
        table[glyph, GlyphGroup.ZAP] = 1
        id_ = num_nle_ids + (glyph - nh.GLYPH_ZAP_OFF) // 4
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(id_, n_bit = DIM_SPECIFIC)

    num_nle_ids += nh.NUM_ZAP

    # (2541, 'GLYPH_SWALLOW_OFF')
    for glyph in range(nh.GLYPH_SWALLOW_OFF, nh.GLYPH_WARNING_OFF):
        table[glyph, GlyphGroup.SWALLOW] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(num_nle_ids, n_bit=DIM_SPECIFIC)

    num_nle_ids += 1

    # (5589, 'GLYPH_WARNING_OFF')
    for glyph in range(nh.GLYPH_WARNING_OFF, nh.GLYPH_STATUE_OFF):
        table[glyph, GlyphGroup.WARNING] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(num_nle_ids, n_bit=DIM_SPECIFIC)
        num_nle_ids += 1

    # (5595, 'GLYPH_STATUE_OFF'), 381 elements of monsters
    for glyph in range(nh.GLYPH_STATUE_OFF, nh.MAX_GLYPH):
        table[glyph, GlyphGroup.STATUE] = 1
        table[glyph, DIM_GLYPH_GROUP + DIM_OBJECT_CLASS:] = number_to_binary(glyph - nh.GLYPH_STATUE_OFF, n_bit=DIM_SPECIFIC)

    # last glyph is additional feature which means no_glyph
    #table[5976] = -np.ones(table[5976].shape, dtype=np.int8)

    table = table.astype(np.bool)

    return table