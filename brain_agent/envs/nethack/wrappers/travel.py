
import gym
from nle.nethack import ACTIONS, Command, CompassCardinalDirection, MiscDirection, MiscAction
from brain_agent.utils.utils import log


TRAVEL_INDEX = ACTIONS.index(Command.TRAVEL)
MOVE_EAST_INDEX = ACTIONS.index(CompassCardinalDirection.E)
MOVE_WEST_INDEX = ACTIONS.index(CompassCardinalDirection.W)
MOVE_NORTH_INDEX = ACTIONS.index(CompassCardinalDirection.N)
MOVE_SOUTH_INDEX = ACTIONS.index(CompassCardinalDirection.S)
WAIT_INDEX = ACTIONS.index(MiscDirection.WAIT)
AGENT_POSITION_INDEX = ACTIONS.index(Command.AUTOPICKUP)
MORE_INDEX = ACTIONS.index(MiscAction.MORE)
ESC_INDEX = ACTIONS.index(Command.ESC)


def print_chars(chars, with_number: bool = True, width=79, height=21):
    """
    Print chars like below.

      |0123456789012345678901234567890123456789012345678901234567890123456789012345678
    --+-------------------------------------------------------------------------------
     0|
     1|                                     ------    -----------
     2|                                     |...[.#   ..........|
     3|                                   ##.=...|# ##|.........|
     4|                                   # |....| ###|....:....|
     5|                                   # ------   #|........)|
     6|                                   #          #..........|
     7|                                   #           -------.---
     8|                                   #                  o
     9|                                   #                  @
    10|                                   #                  u
    11|                                   %                  +
    12|                                ---#
    13|                                ..-#
    14|                                ..|
    15|                          |.......|
    16|                          |<......|
    17|                          ---------
    18|
    19|
    20|

    """
    if with_number:
        print('  |' + ''.join([str(x % 10) for x in range(width)]))
        print('--+' + ('-' * width))
    for i in range(chars.shape[0]):
        if with_number:
            print(f'{i:2d}|', end='')
        print(chars[i].tostring().decode())
    print()


def travel(env, ty, tx):
    """
    travel to (ty, tx)
    """

    # assert 0 < ty < 20, f"current ty: {ty}"
    # assert 0 < tx < 78, f"current tx: {tx}"
    ty = min(max(0, ty), 20)
    tx = min(max(0, tx), 78)

    # Clear previous state
    obs, rew, done, info = env.step(ESC_INDEX)

    # Execute TRAVEL command
    if not done:
        obs, rew, done, info = env.step(TRAVEL_INDEX)

    # Move cursor to current position of agent
    if not done:
        obs, rew, done, info = env.step(AGENT_POSITION_INDEX)

    # Calculate Manhattan distance
    cx, cy = obs['blstats'][:2]
    dy, dx = ty - cy, tx - cx

    # Move cursor to target position (ty, tx)
    for _ in range(abs(dx)):
        if not done:
            obs, rew, done, info = env.step(MOVE_EAST_INDEX if dx > 0 else MOVE_WEST_INDEX)

    for _ in range(abs(dy)):
        if not done:
            obs, rew, done, info = env.step(MOVE_SOUTH_INDEX if dy > 0 else MOVE_NORTH_INDEX)

    if not done:
        # Move agent to cursor position
        obs, rew, done, info = env.step(WAIT_INDEX)

    x, y = obs['blstats'][:2]
    if y != ty or x != tx:
        # print_chars(obs["tty_chars"])
        tty_message = obs["tty_chars"][0].tostring().decode(errors='ignore')
        log.debug(f'cannot move ({cy},{cx}) -> ({ty},{tx}) chars[{ty},{tx}]=\'{chr(obs["chars"][ty, tx])}\' '
                  f'tty message=\"{tty_message}\" misc={obs["misc"]}')
    return obs, rew, done, info


def find_downstair(chars):
    for y in range(chars.shape[0]):
        for x in range(chars.shape[1]):
            if chr(chars[y, x]) == '>':
                return y, x
    return None, None


""" Example 
    def step(self, action):
        obs, r, done, info = self.env_aicrowd.step(action)

        from nle.nethack import ACTIONS, USEFUL_ACTIONS, GLYPH_CMAP_OFF, action_id_to_type, Command, MiscDirection, MiscAction
        from brain_agent.envs.nethack.wrappers.travel import TravelWrapperExample, find_downstair, travel, print_chars
        ty, tx = find_downstair(obs['chars'])
        if tx is not None and ty is not None:
            print_chars(obs['chars'])
            obs, rew, done, info = travel(self.env_aicrowd, ty, tx)
            print_chars(obs['chars'])
            obs, rew, done, info = self.env_aicrowd.step(ACTIONS.index(MiscDirection.DOWN))
            obs, rew, done, info = self.env_aicrowd.step(ACTIONS.index(MiscAction.MORE))
            print(obs['message'].tostring().decode(errors='ignore'))
            print_chars(obs['chars'])
            # raise
"""
