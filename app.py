from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO
from flasgger import Swagger

import os
import json
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')

app = Flask(__name__)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Airbnb Rental Price Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///listings.db'
db = SQLAlchemy(app)

# Define a database model
class Account(db.Model):
    leagueId = db.Column(db.String, primary_key=True)
    tier = db.Column(db.String, nullable=False)
    rank = db.Column(db.String, nullable=False)
    leaguePoints = db.Column(db.Integer, nullable=False)
    wins = db.Column(db.Integer, nullable=False)
    losses = db.Column(db.Integer, nullable=False)
    veteran = db.Column(db.Boolean, nullable=False)
    inactive = db.Column(db.Boolean, nullable=False)
    freshBlood = db.Column(db.Boolean, nullable=False)
    hotStreak = db.Column(db.Boolean, nullable=False)

# Create the database
with app.app_context():
    db.create_all()

def preprocess_data(df):
    # Drop rows where any of the key fields are NaN
    df = df.dropna(subset=['leagueId', 'tier','rank','leaguePoints','wins','losses','veteran','inactive','freshBlood','hotStreak'])

    # One more time, fill any missing numerical values with the median, just in case
    df['wins'] = df['wins'].fillna(df['wins'].median())
    df['losses'] = df['losses'].fillna(df['losses'].median())

    #change rank to numeric
    df['rank'] = df['rank'].replace({'I': 0, 'II': 100, 'III': 200, 'IV': 300}).astype(int)

    #change tier to numeric
    df['tier'] = df['tier'].str.lower().replace({'iron': 0, 'bronze': 400, 'silver': 800, 'gold': 1200, 'platinum': 1600, 'emerald': 2000, 'diamond': 2400}).astype(int)

    #Add total rank column
    df['totalRank'] = df['tier'] + df['rank'] + df['leaguePoints']

    # Drop any rows that still have NaN values at this point (forcefully)
    df = df.dropna()
    return df

# Global variables for model
model = None

@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Reload data from the League of Legends dataset, clear the database, load new data, and return summary stats
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    '''
    global model

    # # Get Data from Riot API (will need to update this to collect a series of data from different ranks)
    league_url = 'https://na1.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/EMERALD/I?page=1&api_key=' + API_TOKEN

    league_response = requests.get(league_url)

    encoding = league_response.headers.get('Content-Encoding')
    content = league_response.content

    if encoding == 'gzip':
        try:
            with gzip.GzipFile(fileobj=BytesIO(content)) as f:
                decompressed_data = f.read()
        except gzip.BadGzipFile:
            print("Error: The file is not gzipped.")
            decompressed_data = content  # Use the original content if decompression fails
    else:
        decompressed_data = content

    accounts = json.loads(decompressed_data)

    seen_ids = set()
    cleaned_accounts = []

    for row in accounts:
        league_id = row.get("leagueId")  # Safely get the leagueId
        if league_id and league_id not in seen_ids:
            cleaned_accounts.append(row)
            seen_ids.add(league_id)

    accounts = cleaned_accounts  # Remove duplicates

    accounts = pd.DataFrame(accounts)

    # Step 3: Clear the database
    db.session.query(Account).delete()

    # Step 4: Process data and insert it into the database
    accounts = accounts[['leagueId', 'tier','rank','leaguePoints','wins','losses','veteran','inactive','freshBlood','hotStreak']].dropna()

    for _, row in accounts.iterrows():
        new_listing = Account(
            leagueId=row['leagueId'],
            tier=row['tier'],
            rank=row['rank'],
            leaguePoints=row['leaguePoints'],
            wins=int(row['wins']),
            losses=int(row['losses']),
            veteran=row['veteran'],
            inactive=row['inactive'],
            freshBlood=row['freshBlood'],
            hotStreak=row['hotStreak'],
        )
        db.session.add(new_listing)
    db.session.commit()

    # Step 5: Preprocess and train model
    df = preprocess_data(accounts) #look at rank since we are only pulling data on emerald accounts
    X = df[['wins', 'losses']]
    y = df['totalRank']
    model = LinearRegression()
    model.fit(X, y)

    summary = {
        'total_accounts': int(len(accounts)),
        'average_wins': float(accounts['wins'].mean()),
        'min_wins': int(accounts['wins'].min()),
        'max_wins': int(accounts['wins'].max()),
        'average_losses': float(accounts['losses'].mean()),
        'min_losses': int(accounts['losses'].min()),
        'max_losses': int(accounts['losses'].max()),
    }

    return jsonify(summary)
@app.route('/predict', methods=['POST']) #Give the predict # wins and # losses and have it predict rank
def predict():
    '''
    Predict the rank of a player based on wins and losses
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            wins:
              type: integer
            losses:
              type: integer
    responses:
      200:
        description: Predicted rank
    '''
    global model  # Ensure that the encoder and model are available for prediction

    # Check if the model is initialized
    if model is None:
        return jsonify({"error": "The data has not been loaded. Please refresh the data by calling the '/reload' endpoint first."}), 400

    data = request.json
    try:
        wins = pd.to_numeric(data.get('wins'), errors='coerce')
        losses = pd.to_numeric(data.get('losses'), errors='coerce')

        if None in [wins, losses]:
            return jsonify({"error": "Missing or invalid required parameters"}), 400

        # Check for NaN values in the converted inputs
        if pd.isna(wins) or pd.isna(losses):
            return jsonify({"error": "Invalid numeric values for wins, or losses"}), 400

        input_data = np.array([[wins, losses]])

        # Predict the price
        predicted_rank = model.predict(input_data)[0]

        return jsonify({"predicted_rank": convert_rank(predicted_rank)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def convert_rank(raw_rank): #this was done with chat gpt
    # Define tiers
    tiers = ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Emerald", "Diamond"]

    # Get tier index
    tier_index = int(raw_rank // 400)
    if tier_index >= len(tiers):  # Cap at Diamond
        tier_index = len(tiers) - 1

    tier = tiers[tier_index]  # Get tier name

    # Get rank within the tier (1-4)
    rank_within_tier = 4 - ((raw_rank % 400) // 100) # Converts 0-99 to IV, 100-199 to III, etc.

    # Get League Points (LP)
    league_points = raw_rank % 100  # The remainder after dividing by 100

    return f"{tier} {int(rank_within_tier)} {int(league_points)} LP"

if __name__ == '__main__':
    app.run(debug=True)
