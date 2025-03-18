from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO
from flasgger import Swagger

import csv
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
    df['rank'] = df['rank'].replace({'I': 1, 'II': 2, 'III': 3, 'IV': 4}).astype(int)

    # One-hot encode the 'neighbourhood_cleansed' column
    # encoder = OneHotEncoder(sparse_output=False)
    # neighbourhood_encoded = encoder.fit_transform(df[['neighbourhood_cleansed']])

    # Create a DataFrame for the one-hot encoded neighborhoods
    # neighbourhood_encoded_df = pd.DataFrame(neighbourhood_encoded, columns=encoder.get_feature_names_out(['neighbourhood_cleansed']))

    # Concatenate the encoded neighborhood with the original dataframe
    # df = pd.concat([df, neighbourhood_encoded_df], axis=1).drop(columns=['neighbourhood_cleansed'])

    # Drop any rows that still have NaN values at this point (forcefully)
    df = df.dropna()
    return df, encoder

# Global variables for model and encoder
model = None
encoder = None

@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Reload data from the League of Legends dataset, clear the database, load new data, and return summary stats
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    '''
    global model, encoder

    # # Step 1: Download and decompress data
    # url = 'https://data.insideairbnb.com/united-states/ma/boston/2024-06-22/data/listings.csv.gz'
    # response = requests.get(url)
    # compressed_file = BytesIO(response.content)
    # decompressed_file = gzip.GzipFile(fileobj=compressed_file)

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

    # Step 2:
    # listings = pd.read_csv(decompressed_file)
    seen_ids = set()
    cleaned_accounts = []

    for row in accounts:
        if row["leagueId"] not in seen_ids:
            cleaned_accounts.append(row)
            seen_ids.add(row["leagueId"])

    accounts = cleaned_accounts  # Remove duplicates


    accounts = pd.DataFrame(accounts)

    # Step 3: Clear the database
    # db.session.query(Listing).delete()
    db.session.query(Account).delete()

    # Step 4: Process data and insert it into the database
    # listings = listings[['price', 'bedrooms', 'bathrooms', 'accommodates', 'neighbourhood_cleansed']].dropna()
    # listings['price'] = listings['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
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


    # for _, row in listings.iterrows():
    #     new_listing = Listing(
    #         price=row['price'],
    #         bedrooms=int(row['bedrooms']),
    #         bathrooms=row['bathrooms'],
    #         accommodates=int(row['accommodates']),
    #         neighbourhood=row['neighbourhood_cleansed']
    #     )
    #     db.session.add(new_listing)
    # db.session.commit()

    # Step 5: Preprocess and train model
    # df, encoder = preprocess_data(listings)
    # X = df.drop(columns='price')
    # y = df['price']
    # model = LinearRegression()
    # model.fit(X, y)

    df, encoder = preprocess_data(accounts) #look at rank since we are only pulling data on emerald accounts
    X = df.drop(columns=['rank', 'leagueId', 'tier', 'freshBlood', 'hotStreak', 'veteran', 'inactive'])
    y = df['rank']
    model = LinearRegression()
    model.fit(X, y)

    # Step 6: Generate summary statistics
    # summary = {
    #     'total_listings': len(listings),
    #     'average_price': listings['price'].mean(),
    #     'min_price': listings['price'].min(),
    #     'max_price': listings['price'].max(),
    #     'average_bedrooms': listings['bedrooms'].mean(),
    #     'average_bathrooms': listings['bathrooms'].mean(),
    #     'top_neighbourhoods': listings['neighbourhood_cleansed'].value_counts().head().to_dict()
    # }

    summary = {
        'total_accounts': int(len(accounts)),
        'average_wins': float(accounts['wins'].mean()),
        'min_wins': int(accounts['wins'].min()),
        'max_wins': int(accounts['wins'].max()),
    }

    return jsonify(summary)
@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the rental price for an Airbnb listing
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            bedrooms:
              type: integer
            bathrooms:
              type: number
            accommodates:
              type: integer
            neighbourhood_cleansed:
              type: string
    responses:
      200:
        description: Predicted rental price
    '''
    global model, encoder  # Ensure that the encoder and model are available for prediction

    # Define the list of valid neighborhoods
    valid_neighborhoods = [
        "East Boston", "Roxbury", "Beacon Hill", "Back Bay", "North End", "Dorchester",
        "Charlestown", "Jamaica Plain", "Downtown", "South Boston", "Bay Village",
        "Brighton", "West Roxbury", "Roslindale", "South End", "Mission Hill",
        "Fenway", "Allston", "Hyde Park", "West End", "Mattapan", "Leather District",
        "South Boston Waterfront", "Chinatown", "Longwood Medical Area"
    ]

    # Check if the model and encoder are initialized
    if model is None or encoder is None:
        return jsonify({"error": "The data has not been loaded. Please refresh the data by calling the '/reload' endpoint first."}), 400

    data = request.json
    try:
        bedrooms = pd.to_numeric(data.get('bedrooms'), errors='coerce')
        bathrooms = pd.to_numeric(data.get('bathrooms'), errors='coerce')
        accommodates = pd.to_numeric(data.get('accommodates'), errors='coerce')
        neighbourhood = data.get('neighbourhood_cleansed')

        if None in [bedrooms, bathrooms, accommodates, neighbourhood]:
            return jsonify({"error": "Missing or invalid required parameters"}), 400

        # Check if the neighborhood is valid
        if neighbourhood not in valid_neighborhoods:
            return jsonify({"error": f"Invalid neighborhood. Please choose one of the following: {', '.join(valid_neighborhoods)}"}), 400

        # Check for NaN values in the converted inputs
        if pd.isna(bedrooms) or pd.isna(bathrooms) or pd.isna(accommodates):
            return jsonify({"error": "Invalid numeric values for bedrooms, bathrooms, or accommodates"}), 400

        # Transform the input using the global encoder
        neighbourhood_encoded = encoder.transform([[neighbourhood]])
        input_data = np.concatenate(([bedrooms, bathrooms, accommodates], neighbourhood_encoded[0]))
        input_data = input_data.reshape(1, -1)

        # Predict the price
        predicted_price = model.predict(input_data)[0]

        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
