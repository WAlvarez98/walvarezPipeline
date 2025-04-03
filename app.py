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
    'title': 'Which team won?',
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

# Create the database
with app.app_context():
    db.create_all()

def preprocess_data(df):
    # Drop rows where any of the key fields are NaN
    df = df.dropna(subset=['leagueId', 'tier','rank'])
    #change rank to numeric
    df['rank'] = df['rank'].replace({'I': 0, 'II': 100, 'III': 200, 'IV': 300}).astype(int)
    #change tier to numeric
    df['tier'] = df['tier'].str.lower().replace({'iron': 0, 'bronze': 400, 'silver': 800, 'gold': 1200, 'platinum': 1600, 'emerald': 2000, 'diamond': 2400}).astype(int)
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

    db.session.query(Account).delete()
    tierList = ['EMERALD', 'DIAMOND']
    rankList = ['I', 'II', 'III', 'IV']

    # seen_ids = set()
    # cleaned_accounts = []

    accountList = []

    for tier in tierList: #need to paginate as these only get first 200 accounts from each rank. Ask professor how to know how much data I need for this project? What is a significant amount of data
        for rank in rankList:

            league_url = 'https://na1.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/'+tier+'/'+rank+'?page=1&api_key=' + API_TOKEN

            league_response = requests.get(league_url)
            accountPage = json.loads(league_response.text)
            accountList.extend(accountPage)

            # for row in accountList:
            #     league_id = row.get("leagueId")  # Safely get the leagueId
            #     if league_id and league_id not in seen_ids:
            #         cleaned_accounts.append(row)
            #         seen_ids.add(league_id)
            # accountList = cleaned_accounts  # Remove duplicates


    accountList = pd.DataFrame(accountList)

    # # Step 4: Process data and insert it into the database
    accountList = accountList[['leagueId', 'tier','rank']].dropna()
    accountList = accountList.drop_duplicates(subset="leagueId", keep="first")

    for _, row in accountList.iterrows():
        new_listing = Account(
            leagueId=row['leagueId'],
            tier=row['tier'],
            rank=row['rank'],
        )
        db.session.add(new_listing)
    db.session.commit()

    return jsonify()

def convert_rank(raw_rank):
    # Define tiers
    tiers = ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Emerald", "Diamond"]

    # Get tier index
    tier_index = int(raw_rank // 400)
    if tier_index >= len(tiers):  # Cap at Diamond
        tier_index = len(tiers) - 1

    tier = tiers[tier_index]  # Get tier name

    # Get rank within the tier (1-4)
    rank_within_tier = 4 - ((raw_rank % 400) // 100) # Converts 0-99 to IV, 100-199 to III, etc.

    return f"{tier} {int(rank_within_tier)}"

if __name__ == '__main__':
    app.run(debug=True)
