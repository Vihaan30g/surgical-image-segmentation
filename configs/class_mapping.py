
CLASS_INFO = {

    0: {
        "name": "background",
        "raw_value": 80
    },

    1: {
        "name": "abdominal_wall",
        "raw_value": 17
    },

    2: {
        "name": "liver",
        "raw_value": 33
    },

    3: {
        "name": "gastrointestinal_tract",
        "raw_value": 19
    },

    4: {
        "name": "fat",
        "raw_value": 18
    },

    5: {
        "name": "grasper",
        "raw_value": 49
    },

    6: {
        "name": "connective_tissue",
        "raw_value": 35
    },

    7: {
        "name": "blood",
        "raw_value": 36
    },

    8: {
        "name": "cystic_duct",
        "raw_value": 37
    },

    9: {
        "name": "l_hook_electrocautery",
        "raw_value": 50
    },

    10: {
        "name": "gallbladder",
        "raw_value": 34
    },

    11: {
        "name": "hepatic_vein",
        "raw_value": 51
    },

    12: {
        "name": "liver_ligament",
        "raw_value": 5
    }
}


# RAW WATERSHED VALUE -> TRAIN CLASS ID
RAW_TO_CLASS = {
    info["raw_value"]: class_id
    for class_id, info in CLASS_INFO.items()
}


# TRAIN CLASS ID -> CLASS NAME
CLASS_NAMES = {
    class_id: info["name"]
    for class_id, info in CLASS_INFO.items()
}
