import json
import os
import random

def create_training_weights(filename):
    """
    Generate training data with random weights for different parameters.

    Args:
        filename (str): The name of the file to save the training data.

    Returns:
        None
    """
    weights = {
        "Level of Detail": {
            "Background": 0.75,
            "Foreground": 1,
            "Hero": 1.25
        },
        "Type": {
            "Shot": 0.75,
            "Dev": 1,
            "Dev+Shot": 1.25
        },
        "Artist Level": {
            "Junior": 1.5,
            "Mid": 0.75,
            "Senior": 0.67,
            "Supervisor": 0.32
        },
        "Tag": {
            "explosion": 5,
            "spark": 1,
            "water": 10,
            "smoke": 3
        }
    }

    examples = []

    for _ in range(1000):
        level_of_detail = random.choice(list(weights["Level of Detail"].keys()))
        type_ = random.choice(list(weights["Type"].keys()))
        artist_level = random.choice(list(weights["Artist Level"].keys()))
        tags = random.sample(list(weights["Tag"].keys()), random.randint(1, 3))
        bid_days = sum([weights["Tag"][tag] for tag in tags]) * weights["Level of Detail"][level_of_detail] * weights["Type"][type_] * weights["Artist Level"][artist_level]
        
        example = {
            "Level of Detail": level_of_detail,
            "Type": type_,
            "Artist Level": artist_level,
            "Tag": tags,
            "Bid Days": bid_days
        }
        examples.append(example)

    with open(os.path.join('training_data', filename), 'w') as f:
        json.dump(examples, f, indent=4)

# Usage
create_training_weights('basic.json')