DMLAB_INSTRUCTIONS = 'INSTR'
DMLAB_VOCABULARY_SIZE = 1000
DMLAB_MAX_INSTRUCTION_LEN = 16


HUMAN_SCORES = {
    'rooms_collect_good_objects_test': 10,
    'rooms_exploit_deferred_effects_test': 85.65,
    'rooms_select_nonmatching_object': 65.9,
    'rooms_watermaze': 54,
    'rooms_keys_doors_puzzle': 53.8,
    'language_select_described_object': 389.5,
    'language_select_located_object': 280.7,
    'language_execute_random_task': 254.05,
    'language_answer_quantitative_question': 184.5,
    'lasertag_one_opponent_small': 12.65,
    'lasertag_three_opponents_small': 18.55,
    'lasertag_one_opponent_large': 18.6,
    'lasertag_three_opponents_large': 31.5,
    'natlab_fixed_large_map': 36.9,
    'natlab_varying_map_regrowth': 24.45,
    'natlab_varying_map_randomized': 42.35,
    'skymaze_irreversible_path_hard': 100,
    'skymaze_irreversible_path_varied': 100,
    'psychlab_arbitrary_visuomotor_mapping': 58.75,
    'psychlab_continuous_recognition': 58.3,
    'psychlab_sequential_comparison': 39.5,
    'psychlab_visual_search': 78.5,
    'explore_object_locations_small': 74.45,
    'explore_object_locations_large': 65.65,
    'explore_obstructed_goals_small': 206,
    'explore_obstructed_goals_large': 119.5,
    'explore_goal_locations_small': 267.5,
    'explore_goal_locations_large': 194.5,
    'explore_object_rewards_few': 77.7,
    'explore_object_rewards_many': 106.7,
}


RANDOM_SCORES = {
    'rooms_collect_good_objects_test': 0.073,
    'rooms_exploit_deferred_effects_test': 8.501,
    'rooms_select_nonmatching_object': 0.312,
    'rooms_watermaze': 4.065,
    'rooms_keys_doors_puzzle': 4.135,
    'language_select_described_object': -0.07,
    'language_select_located_object': 1.929,
    'language_execute_random_task': -5.913,
    'language_answer_quantitative_question': -0.33,
    'lasertag_one_opponent_small': -0.224,
    'lasertag_three_opponents_small': -0.214,
    'lasertag_one_opponent_large': -0.083,
    'lasertag_three_opponents_large': -0.102,
    'natlab_fixed_large_map': 2.173,
    'natlab_varying_map_regrowth': 2.989,
    'natlab_varying_map_randomized': 7.346,
    'skymaze_irreversible_path_hard': 0.1,
    'skymaze_irreversible_path_varied': 14.4,
    'psychlab_arbitrary_visuomotor_mapping': 0.163,
    'psychlab_continuous_recognition': 0.224,
    'psychlab_sequential_comparison': 0.129,
    'psychlab_visual_search': 0.085,
    'explore_object_locations_small': 3.575,
    'explore_object_locations_large': 4.673,
    'explore_obstructed_goals_small': 6.76,
    'explore_obstructed_goals_large': 2.61,
    'explore_goal_locations_small': 7.66,
    'explore_goal_locations_large': 3.14,
    'explore_object_rewards_few': 2.073,
    'explore_object_rewards_many': 2.438,
}

DMLAB_LEVELS_BY_ENVNAME = {
    'dmlab_30':
        ['contributed/dmlab30/rooms_collect_good_objects_train',
         'contributed/dmlab30/rooms_exploit_deferred_effects_train',
         'contributed/dmlab30/rooms_select_nonmatching_object',
         'contributed/dmlab30/rooms_watermaze',
         'contributed/dmlab30/rooms_keys_doors_puzzle',
         'contributed/dmlab30/language_select_described_object',
         'contributed/dmlab30/language_select_located_object',
         'contributed/dmlab30/language_execute_random_task',
         'contributed/dmlab30/language_answer_quantitative_question',
         'contributed/dmlab30/lasertag_one_opponent_small',
         'contributed/dmlab30/lasertag_three_opponents_small',
         'contributed/dmlab30/lasertag_one_opponent_large',
         'contributed/dmlab30/lasertag_three_opponents_large',
         'contributed/dmlab30/natlab_fixed_large_map',
         'contributed/dmlab30/natlab_varying_map_regrowth',
         'contributed/dmlab30/natlab_varying_map_randomized',
         'contributed/dmlab30/skymaze_irreversible_path_hard',
         'contributed/dmlab30/skymaze_irreversible_path_varied',
         'contributed/dmlab30/psychlab_arbitrary_visuomotor_mapping',
         'contributed/dmlab30/psychlab_continuous_recognition',
         'contributed/dmlab30/psychlab_sequential_comparison',
         'contributed/dmlab30/psychlab_visual_search',
         'contributed/dmlab30/explore_object_locations_small',
         'contributed/dmlab30/explore_object_locations_large',
         'contributed/dmlab30/explore_obstructed_goals_small',
         'contributed/dmlab30/explore_obstructed_goals_large',
         'contributed/dmlab30/explore_goal_locations_small',
         'contributed/dmlab30/explore_goal_locations_large',
         'contributed/dmlab30/explore_object_rewards_few',
         'contributed/dmlab30/explore_object_rewards_many'
         ],
    'dmlab_30_test':
        ['contributed/dmlab30/rooms_collect_good_objects_test',
         'contributed/dmlab30/rooms_exploit_deferred_effects_test',
         'contributed/dmlab30/rooms_select_nonmatching_object',
         'contributed/dmlab30/rooms_watermaze',
         'contributed/dmlab30/rooms_keys_doors_puzzle',
         'contributed/dmlab30/language_select_described_object',
         'contributed/dmlab30/language_select_located_object',
         'contributed/dmlab30/language_execute_random_task',
         'contributed/dmlab30/language_answer_quantitative_question',
         'contributed/dmlab30/lasertag_one_opponent_small',
         'contributed/dmlab30/lasertag_three_opponents_small',
         'contributed/dmlab30/lasertag_one_opponent_large',
         'contributed/dmlab30/lasertag_three_opponents_large',
         'contributed/dmlab30/natlab_fixed_large_map',
         'contributed/dmlab30/natlab_varying_map_regrowth',
         'contributed/dmlab30/natlab_varying_map_randomized',
         'contributed/dmlab30/skymaze_irreversible_path_hard',
         'contributed/dmlab30/skymaze_irreversible_path_varied',
         'contributed/dmlab30/psychlab_arbitrary_visuomotor_mapping',
         'contributed/dmlab30/psychlab_continuous_recognition',
         'contributed/dmlab30/psychlab_sequential_comparison',
         'contributed/dmlab30/psychlab_visual_search',
         'contributed/dmlab30/explore_object_locations_small',
         'contributed/dmlab30/explore_object_locations_large',
         'contributed/dmlab30/explore_obstructed_goals_small',
         'contributed/dmlab30/explore_obstructed_goals_large',
         'contributed/dmlab30/explore_goal_locations_small',
         'contributed/dmlab30/explore_goal_locations_large',
         'contributed/dmlab30/explore_object_rewards_few',
         'contributed/dmlab30/explore_object_rewards_many'
         ]
}

IMPALA_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)


EXTENDED_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-10, 0, 0, 0, 0, 0, 0),  # Small Look Left
    (10, 0, 0, 0, 0, 0, 0),   # Small Look Right
    (-60, 0, 0, 0, 0, 0, 0),  # Large Look Left
    (60, 0, 0, 0, 0, 0, 0),   # Large Look Right
    (0, 10, 0, 0, 0, 0, 0),   # Look Down
    (0, -10, 0, 0, 0, 0, 0),  # Look Up
    (-10, 0, 0, 1, 0, 0, 0),  # Forward + Small Look Left
    (10, 0, 0, 1, 0, 0, 0),   # Forward + Small Look Right
    (-60, 0, 0, 1, 0, 0, 0),  # Forward + Large Look Left
    (60, 0, 0, 1, 0, 0, 0),   # Forward + Large Look Right
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)

EXTENDED_ACTION_SET_LARGE = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-10, 0, 0, 0, 0, 0, 0),  # Small Look Left
    (10, 0, 0, 0, 0, 0, 0),   # Small Look Right
    (-60, 0, 0, 0, 0, 0, 0),  # Large Look Left
    (60, 0, 0, 0, 0, 0, 0),   # Large Look Right
    (0, 10, 0, 0, 0, 0, 0),   # Look Down
    (0, -10, 0, 0, 0, 0, 0),  # Look Up
    (0, 60, 0, 0, 0, 0, 0),   # Large Look Down
    (0, -60, 0, 0, 0, 0, 0),  # Large Look Up
    (-10, 0, 0, 1, 0, 0, 0),  # Forward + Small Look Left
    (10, 0, 0, 1, 0, 0, 0),   # Forward + Small Look Right
    (-60, 0, 0, 1, 0, 0, 0),  # Forward + Large Look Left
    (60, 0, 0, 1, 0, 0, 0),   # Forward + Large Look Right
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)
