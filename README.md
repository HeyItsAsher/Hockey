# NHL Game Predictor

This Python script predicts the outcome of NHL games based on team statistics using machine learning.

## Features

- Fetches real-time team statistics from the NHL API
- Uses various team performance metrics including:
  - Points per game
  - Goals for/against
  - Faceoff win percentage
  - Shooting percentage
  - Save percentage
  - Power play percentage
  - Penalty kill percentage
- Employs a Random Forest Classifier for predictions
- Provides win probabilities for both teams

## Requirements

- Python 3.7+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Simply run the script:
```bash
python nhl_game_predictor.py
```

The script will:
1. Fetch current NHL team statistics
2. Train a model using the available data
3. Show an example prediction between two teams

## How it Works

The predictor:
1. Fetches current season statistics for all NHL teams
2. Creates a synthetic dataset based on team performance metrics
3. Trains a Random Forest model on the data
4. Uses the trained model to predict game outcomes

The prediction is based on various team statistics and takes into account home/away team advantages.

## Note

This is a statistical model and predictions are based on historical performance metrics. Many factors can influence the actual outcome of a game that aren't captured in these statistics. 