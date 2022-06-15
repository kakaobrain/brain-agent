# When a piece of armor is enchanted, the enchantment is added to the armor's AC rating.
ArmorData = {
    'Hawaiian shirt': {'Type': 'Shirts', 'Cost': 3, 'Weight': 5, 'AC': 0},
    'T-shirt': {'Type': 'Shirts', 'Cost': 2, 'Weight': 5, 'AC': 0},
    'T shirt': {'Type': 'Shirts', 'Cost': 2, 'Weight': 5, 'AC': 0},
    'plate mail': {'Type': 'Suits', 'Cost': 600, 'Weight': 450, 'AC': 7},
    'tanko': {'Type': 'Suits', 'Cost': 600, 'Weight': 450, 'AC': 7}, # samurai
    'bronze plate mail': {'Type': 'Suits', 'Cost': 400, 'Weight': 450, 'AC': 6},
    'splint mail': {'Type': 'Suits', 'Cost': 80, 'Weight': 400, 'AC': 6},
    'banded mail': {'Type': 'Suits', 'Cost': 90, 'Weight': 350, 'AC': 6},
    'dwarvish mithril-coat': {'Type': 'Suits', 'Cost': 240, 'Weight': 150, 'AC': 6},
    'dwarvish mithril coat': {'Type': 'Suits', 'Cost': 240, 'Weight': 150, 'AC': 6},
    'elven mithril-coat': {'Type': 'Suits', 'Cost': 240, 'Weight': 150, 'AC': 5},
    'elven mithril coat': {'Type': 'Suits', 'Cost': 240, 'Weight': 150, 'AC': 5},
    'chain mail': {'Type': 'Suits', 'Cost': 75, 'Weight': 300, 'AC': 5},
    'orcish chain mail': {'Type': 'Suits', 'Cost': 75, 'Weight': 300, 'AC': 4},
    'crude chain mail': {'Type': 'Suits', 'Cost': 75, 'Weight': 300, 'AC': 4},
    'scale mail': {'Type': 'Suits', 'Cost': 45, 'Weight': 250, 'AC': 4},
    'studded leather armor': {'Type': 'Suits', 'Cost': 15, 'Weight': 200, 'AC': 3},
    'ring mail': {'Type': 'Suits', 'Cost': 100, 'Weight': 250, 'AC': 3},
    'orcish ring mail': {'Type': 'Suits', 'Cost': 80, 'Weight': 250, 'AC': 2},
    'crude ring mail': {'Type': 'Suits', 'Cost': 80, 'Weight': 250, 'AC': 2},
    'leather armor': {'Type': 'Suits', 'Cost': 5, 'Weight': 150, 'AC': 2},
    'leather jacket': {'Type': 'Suits', 'Cost': 10, 'Weight': 30, 'AC': 1},
    'tanko': {'Type': 'Suits', 'Cost': 600, 'Weight': 450, 'AC': 7},
    'plate mail (tanko)': {'Type': 'Suits', 'Cost': 600, 'Weight': 450, 'AC': 7},
    'crystal plate mail': {'Type': 'Suits', 'Cost': 820, 'Weight': 450, 'AC': 7},
    'dragon scales': {'Type': 'Suits', 'Cost': 500, 'Weight': 40, 'AC': 3},
    'dragon scale mail': {'Type': 'Suits', 'Cost': 900, 'Weight': 40, 'AC': 9},
    'mummy wrapping': {'Type': 'Cloaks', 'Cost': 2, 'Weight': 3, 'AC': 0},
    'elven cloak': {'Type': 'Cloaks', 'Cost': 60, 'Weight': 10, 'AC': 1},
    'faded pall': {'Type': 'Cloaks', 'Cost': 60, 'Weight': 10, 'AC': 1},
    'orcish cloak': {'Type': 'Cloaks', 'Cost': 40, 'Weight': 10, 'AC': 0},
    'coarse mantelet': {'Type': 'Cloaks', 'Cost': 40, 'Weight': 10, 'AC': 0},
    'dwarvish cloak': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 0},
    'hooded cloak': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 0},
    'oilskin cloak': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'slippery cloak': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'robe': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 15, 'AC': 2},
    'alchemy smock': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'apron': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'leather cloak': {'Type': 'Cloaks', 'Cost': 40, 'Weight': 15, 'AC': 1},
    'cloak of protection': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 3},
    'tattered cape': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 3},
    'cloak of invisibility': {'Type': 'Cloaks', 'Cost': 60, 'Weight': 10, 'AC': 1},
    'opera cloak': {'Type': 'Cloaks', 'Cost': 60, 'Weight': 10, 'AC': 1},
    'cloak of magic resistance': {'Type': 'Cloaks', 'Cost': 60, 'Weight': 10, 'AC': 1},
    'ornamental cope': {'Type': 'Cloaks', 'Cost': 60, 'Weight': 10, 'AC': 1},
    'cloak of displacement': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'piece of cloth': {'Type': 'Cloaks', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'elven leather helm': {'Type': 'Helms', 'Cost': 8, 'Weight': 3, 'AC': 1},
    'leather hat': {'Type': 'Helms', 'Cost': 8, 'Weight': 3, 'AC': 1},
    'orcish helm': {'Type': 'Helms', 'Cost': 10, 'Weight': 30, 'AC': 1},
    'iron skull cap': {'Type': 'Helms', 'Cost': 10, 'Weight': 30, 'AC': 1},
    'dwarvish iron helm': {'Type': 'Helms', 'Cost': 20, 'Weight': 40, 'AC': 2},
    'hard hat': {'Type': 'Helms', 'Cost': 20, 'Weight': 40, 'AC': 2},
    'fedora': {'Type': 'Helms', 'Cost': 1, 'Weight': 3, 'AC': 0},
    'cornuthaum': {'Type': 'Helms', 'Cost': 80, 'Weight': 4, 'AC': 0},
    'conical hat': {'Type': 'Helms', 'Cost': 80, 'Weight': 4, 'AC': 0},
    'dunce cap': {'Type': 'Helms', 'Cost': 1, 'Weight': 4, 'AC': 0},
    'conical hat': {'Type': 'Helms', 'Cost': 1, 'Weight': 4, 'AC': 0},
    'dented pot': {'Type': 'Helms', 'Cost': 8, 'Weight': 10, 'AC': 1},
    'helmet': {'Type': 'Helms', 'Cost': 10, 'Weight': 30, 'AC': 1},
    'kabuto': {'Type': 'Helms', 'Cost': 10, 'Weight': 30, 'AC': 1}, # samurai
    'plumed helmet': {'Type': 'Helms', 'Cost': 10, 'Weight': 30, 'AC': 1},
    'helm of brilliance': {'Type': 'Helms', 'Cost': 50, 'Weight': 50, 'AC': 1},
    'etched helmet': {'Type': 'Helms', 'Cost': 50, 'Weight': 50, 'AC': 1},
    'helm of opposite alignment': {'Type': 'Helms', 'Cost': 50, 'Weight': 50, 'AC': 1},
    'crested helmet': {'Type': 'Helms', 'Cost': 50, 'Weight': 50, 'AC': 1},
    'helm of telepathy': {'Type': 'Helms', 'Cost': 50, 'Weight': 50, 'AC': 1},
    'visored helmet': {'Type': 'Helms', 'Cost': 50, 'Weight': 50, 'AC': 1},
    'helmet (kabuto)': {'Type': 'Helms', 'Cost': 10, 'Weight': 30, 'AC': 1},
    'kabuto': {'Type': 'Helms', 'Cost': 10, 'Weight': 30, 'AC': 1},
    'leather gloves': {'Type': 'Gloves', 'Cost': 8, 'Weight': 10, 'AC': 1},
    'yugake': {'Type': 'Gloves', 'Cost': 8, 'Weight': 10, 'AC': 1}, # samurai
    'old gloves': {'Type': 'Gloves', 'Cost': 8, 'Weight': 10, 'AC': 1},
    'yugake': {'Type': 'Gloves', 'Cost': 8, 'Weight': 10, 'AC': 1},
    'gauntlets of fumbling': {'Type': 'Gloves', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'padded gloves': {'Type': 'Gloves', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'gauntlets of power': {'Type': 'Gloves', 'Cost': 50, 'Weight': 30, 'AC': 1},
    'riding gloves': {'Type': 'Gloves', 'Cost': 50, 'Weight': 30, 'AC': 1},
    'gauntlets of dexterity': {'Type': 'Gloves', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'fencing gloves': {'Type': 'Gloves', 'Cost': 50, 'Weight': 10, 'AC': 1},
    'leather gloves (yugake)': {'Type': 'Gloves', 'Cost': 8, 'Weight': 10, 'AC': 1},
    'small shield': {'Type': 'Shields', 'Cost': 3, 'Weight': 30, 'AC': 1},
    'elven shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 40, 'AC': 2},
    'blue and green shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 40, 'AC': 2},
    'Uruk-hai shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 50, 'AC': 1},
    'Uruk hai shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 50, 'AC': 1},
    'white-handed shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 50, 'AC': 1},
    'white handed shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 50, 'AC': 1},
    'orcish shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 50, 'AC': 1},
    'red-eyed shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 50, 'AC': 1},
    'red eyed shield': {'Type': 'Shields', 'Cost': 7, 'Weight': 50, 'AC': 1},
    'large shield': {'Type': 'Shields', 'Cost': 10, 'Weight': 100, 'AC': 2},
    'dwarvish roundshield': {'Type': 'Shields', 'Cost': 10, 'Weight': 100, 'AC': 2},
    'large round shield': {'Type': 'Shields', 'Cost': 10, 'Weight': 100, 'AC': 2},
    'shield of reflection': {'Type': 'Shields', 'Cost': 50, 'Weight': 50, 'AC': 2},
    'polished silver shield': {'Type': 'Shields', 'Cost': 50, 'Weight': 50, 'AC': 2},
    'low boots': {'Type': 'Boots', 'Cost': 8, 'Weight': 10, 'AC': 1},
    'walking shoes': {'Type': 'Boots', 'Cost': 8, 'Weight': 10, 'AC': 1},
    'iron shoes': {'Type': 'Boots', 'Cost': 16, 'Weight': 50, 'AC': 2},
    'hard shoes': {'Type': 'Boots', 'Cost': 16, 'Weight': 50, 'AC': 2},
    'high boots': {'Type': 'Boots', 'Cost': 12, 'Weight': 20, 'AC': 2},
    'jackboots': {'Type': 'Boots', 'Cost': 12, 'Weight': 20, 'AC': 2},
    'speed boots': {'Type': 'Boots', 'Cost': 50, 'Weight': 20, 'AC': 1},
    'combat boots': {'Type': 'Boots', 'Cost': 50, 'Weight': 20, 'AC': 1},
    'water walking boots': {'Type': 'Boots', 'Cost': 50, 'Weight': 20, 'AC': 1},
    'jungle boots': {'Type': 'Boots', 'Cost': 50, 'Weight': 20, 'AC': 1},
    'jumping boots': {'Type': 'Boots', 'Cost': 50, 'Weight': 20, 'AC': 1},
    'hiking boots': {'Type': 'Boots', 'Cost': 50, 'Weight': 20, 'AC': 1},
    'elven boots': {'Type': 'Boots', 'Cost': 8, 'Weight': 15, 'AC': 1},
    'mud boots': {'Type': 'Boots', 'Cost': 8, 'Weight': 15, 'AC': 1},
    'kicking boots': {'Type': 'Boots', 'Cost': 8, 'Weight': 50, 'AC': 1},
    'buckled boots': {'Type': 'Boots', 'Cost': 8, 'Weight': 50, 'AC': 1},
    'fumble boots': {'Type': 'Boots', 'Cost': 30, 'Weight': 20, 'AC': 1},
    'riding boots': {'Type': 'Boots', 'Cost': 30, 'Weight': 20, 'AC': 1},
    'levitation boots': {'Type': 'Boots', 'Cost': 30, 'Weight': 15, 'AC': 1},
    'snow boots': {'Type': 'Boots', 'Cost': 30, 'Weight': 15, 'AC': 1},
}
# print(ArmorData.get('iron shoes').get('AC'))