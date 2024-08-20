
# Movie Recommendation System using LightFM

This project is a simple movie recommendation system built using the [LightFM](https://making.lyst.com/lightfm/docs/home.html) library. It uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) to train a model and then recommends movies to users based on their preferences.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following Python packages installed:

- `numpy`
- `lightfm`

You can install them using pip:

```bash
pip install numpy lightfm
```

### Code Explanation

```python
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format it
data = fetch_movielens(min_rating=4.0)

# Print Training and testing data
print(repr(data['train']))
print(repr(data['test']))

# Create model
model = LightFM(loss='warp')

# Train model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    
    # Number of users and movies in training data
    n_users, n_items = data['train'].shape

    # Generate recommendations for each user we input
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        # Print out the results
        print("User %s" % user_id)
        print("     Known positives:")
        for x in known_positives[:3]:
            print("         %s" % x)
        print("     Recommended:")
        for x in top_items[:3]:
            print("     %s" % x)

# Sample recommendations for specific users
sample_recommendation(model, data, [3, 25, 450])
```

### How to Run

1. Ensure that you have all the required libraries installed.
2. Copy the code above into a Python file (e.g., `recommendation_system.py`).
3. Run the Python file in your terminal or IDE.
4. The program will fetch the MovieLens dataset, train a recommendation model, and provide movie recommendations for specified users.

