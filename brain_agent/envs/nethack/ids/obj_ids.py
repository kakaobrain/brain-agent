from brain_agent.utils.utils import AttrDict
import nle.nethack as nh
from brain_agent.envs.nethack.ids.weapon import WeaponData
from brain_agent.envs.nethack.ids.armor import ArmorData
import numpy as np

GlyphGroup = AttrDict(
    # See display.h in NetHack.
    MON = 0,
    PET = 1,
    INVIS = 2,
    DETECT = 3,
    BODY = 4,
    RIDDEN = 5,
    OBJ = 6,
    CMAP = 7,
    EXPLODE = 8,
    ZAP = 9,
    SWALLOW = 10,
    WARNING = 11,
    STATUE = 12,
)

ObjClasses = AttrDict(
    RANDOM = 0,
    ILLOBJ = 1,
    WEAPON = 2,
    ARMOR = 3,
    RING = 4,
    AMULET = 5,
    TOOL = 6,
    FOOD = 7,
    POTION = 8,
    SCROLL = 9,
    SPBOOK = 10,
    WAND = 11,
    COIN = 12,
    GEM = 13,
    ROCK = 14,
    BALL = 15,
    CHAIN = 16,
    VENOM = 17,
    MAXOCLASSES = 18,
)

ObjNames = [nh.OBJ_NAME(nh.objclass(idx)) for idx in range(nh.NUM_OBJECTS) if not nh.OBJ_NAME(nh.objclass(idx)) is None]

INDEXED_ARMOR_DATA = AttrDict()
INDEXED_WEAPON_DATA = AttrDict()

def convert_armor_type_to_onehot(type):
    if type == 'Shirts':
        return [1, 0, 0, 0, 0, 0, 0]
    elif type == 'Suits':
        return [0, 1, 0, 0, 0, 0, 0]
    elif type == 'Cloaks':
        return [0, 0, 1, 0, 0, 0, 0]
    elif type == 'Helms':
        return [0, 0, 0, 1, 0, 0, 0]
    elif type == 'Gloves':
        return [0, 0, 0, 0, 1, 0, 0]
    elif type == 'Shields':
        return [0, 0, 0, 0, 0, 1, 0]
    elif type == 'Boots':
        return [0, 0, 0, 0, 0, 0, 1]

def convert_weapon_type_to_onehot(type):
    all_types = ['dagger', 'knife', 'axe', 'pick-axe', 'short sword', 'broadsword', 'long sword', 'two-handed sword',
                 'scimitar', 'saber', 'club', 'mace', 'morning star', 'flail', 'hammer', 'quarterstaff', 'polearms',
                 'spear', 'trident', 'lance', 'bow', 'sling', 'crossbow', 'dart', 'shuriken', 'boomerang', 'whip',
                 'unicorn horn', ]
    val = [0]*len(all_types)
    val[all_types.index(type)] = 1
    tmp = np.array(val)
    return val

for idx in range(nh.NUM_OBJECTS):
    obj_name = nh.OBJ_NAME(nh.objclass(idx))
    if ArmorData.get(obj_name, None) is not None:
        armor_data = ArmorData[obj_name]
        data = convert_armor_type_to_onehot(armor_data['Type'])
        data.append(armor_data['Cost']/100)
        data.append(armor_data['Weight']/100)
        data.append(armor_data['AC'])
        INDEXED_ARMOR_DATA[idx+nh.GLYPH_OBJ_OFF] = AttrDict(name=obj_name, feature=np.array(data))

    if WeaponData.get(obj_name, None) is not None:
        weapon_data = WeaponData[obj_name]
        data = convert_weapon_type_to_onehot(weapon_data['Type'])
        data.append(weapon_data['Cost']/100)
        data.append(weapon_data['Weight']/100)
        data.append(weapon_data['Damage_small'])
        data.append(weapon_data['Damage_large'])
        data.append(weapon_data['Damage'])
        INDEXED_WEAPON_DATA[idx+nh.GLYPH_OBJ_OFF] = AttrDict(name=obj_name, feature=np.array(data))

# print(INDEXED_WEAPON_DATA[1961])

# idx = 1961-nh.GLYPH_OBJ_OFF
# print (nh.OBJ_NAME(nh.objclass(idx)))
