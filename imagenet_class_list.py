''' 27 '''
insect = [
        "cricket",
        "ant",
        "leafhopper",
        "walking_stick",
        "grasshopper",
        "dung_beetle",
        "tiger_beetle",
        "lacewing",
        "rhinoceros_beetle",
        "ringlet",
        "long-horned_beetle",
        "ladybug",
        "ground_beetle",
        "cicada",
        "cabbage_butterfly",
        "leaf_beetle",
        "lycaenid",
        "bee",
        "monarch",
        "damselfly",
        "admiral",
        "sulphur_butterfly",
        "dragonfly",
        "fly",
        "weevil",
        "cockroach",
        "mantis",]
''' 26 '''
terrier = [
        "Lakeland_terrier",
        "Scotch_terrier",
        "cairn",
        "Airedale",
        "Tibetan_terrier",
        "Yorkshire_terrier",
        "Norfolk_terrier",
        "Staffordshire_bullterrier",
        "Sealyham_terrier",
        "standard_schnauzer",
        "Norwich_terrier",
        "Bedlington_terrier",
        "Lhasa",
        "Irish_terrier",
        "silky_terrier",
        "Dandie_Dinmont",
        "Boston_bull",
        "Border_terrier",
        "soft-coated_wheaten_terrier",
        "Australian_terrier",
        "American_Staffordshire_terrier",
        "West_Highland_white_terrier",
        "giant_schnauzer",
        "miniature_schnauzer",
        "Kerry_blue_terrier",
        "wire-haired_fox_terrier",]

''' 16 '''
fish = [
        "tench",
        "stingray",
        "tiger_shark",
        "barracouta",
        "coho",
        "gar",
        "electric_ray",
        "great_white_shark",
        "sturgeon",
        "puffer",
        "anemone_fish",
        "goldfish",
        "eel",
        "rock_beauty",
        "lionfish",
        "hammerhead"]
''' 11 '''
lizard = [
        "agama",
        "banded_gecko",
        "Komodo_dragon",
        "frilled_lizard",
        "African_chameleon",
        "American_chameleon",
        "green_lizard",
        "whiptail",
        "common_iguana",
        "alligator_lizard",
        "Gila_monster",]
''' 9 '''
truck =["pickup", 
        "police_van",
        "trailer_truck",
        "minivan",
        "moving_van",
        "tow_truck",
        "fire_engine",
        "garbage_truck",
        "tractor"]
automobile = ["beach_wagon",
        "convertible",
        "sports_car",
        "ambulance",
        "jeep",
        "limousine",
        "racer",
        "cab",
        "Model_T"]
def icl(name):
    if name == 'truck':
        return truck
    elif name == 'automobile':
        return automobile
    elif name == 'insect':
        return insect
    elif name == 'terrier':
        return terrier
    elif name == 'fish':
        return fish
    elif name == 'lizard':
        return lizard
    else:
        return None
