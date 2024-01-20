import json
import os
import random
import json

def create_training_weights(filename):
    """
    Generate training data with random weights for different parameters.

    Args:
        filename (str): The name of the file to save the training data.

    Returns:
        None
    """
    with open('examples/basic_weights.json') as f:
        weights = json.load(f)

    examples = []

    for _ in range(10000):
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