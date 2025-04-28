from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import requests

# For rate limiting
import backoff
import time

from flasgger import Swagger

import os
from dotenv import load_dotenv


timeouttime = 0.06

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
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'listings.db')
db = SQLAlchemy(app)


# Define a database model
class Account(db.Model):
    puuid = db.Column(db.String, primary_key=True)
    tier = db.Column(db.String, nullable=False)
    rank = db.Column(db.String, nullable=False)

class Match(db.Model):
    matchId = db.Column(db.String, primary_key=True)
    team1Win = db.Column(db.Boolean, nullable=False)
    champ1  = db.Column(db.String, nullable=False)
    champ2  = db.Column(db.String, nullable=False)
    champ3  = db.Column(db.String, nullable=False)
    champ4  = db.Column(db.String, nullable=False)
    champ5  = db.Column(db.String, nullable=False)
    champ6  = db.Column(db.String, nullable=False)
    champ7  = db.Column(db.String, nullable=False)
    champ8  = db.Column(db.String, nullable=False)
    champ9  = db.Column(db.String, nullable=False)
    champ10  = db.Column(db.String, nullable=False)

# Create the database
with app.app_context():
    db.create_all()

@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_tries=10
)
def make_request(url):
    response = requests.get(url)

    if response.status_code == 429:
        print("Rate limit hit")
        raise requests.exceptions.RequestException("Rate limited")

    if response.status_code != 200:
        raise requests.exceptions.RequestException(f"Bad status: {response.status_code}")

    return response.json()

def fetch_matches(matches_to_search):
    matchesData = []
    print("begin fetch_matches ------------------------------------------------------------")

    for matchId in matches_to_search:
        url = f'https://americas.api.riotgames.com/lol/match/v5/matches/{matchId}?api_key={API_TOKEN}'
        team1champions = []
        team2champions = []
        try:
            matchesPage = make_request(url)
            participants = matchesPage['info']['participants']
            team1Win = matchesPage['info']['teams'][0]['win']
            for participant in participants:
                if participant['teamId'] == 100:
                    team1champions.append(participant['championName'])
                elif participant['teamId'] == 200:
                    team2champions.append(participant['championName'])

            matchesData.append({
                'matchId': matchId,
                'team1Champions': team1champions,
                'team2Champions': team2champions,
                'team1Win': team1Win
            })
        except requests.exceptions.RequestException as e:
            print(f'Failed to get matches for {matchId}: {e}')
            continue
        time.sleep(timeouttime)
    print("end fetch_matches ------------------------------------------------------------")
    return matchesData

#info -> endOfGameResult == GameComplete -> Participants -> 0 -> champion name

def fetch_puuids(): #add saftey to allow me to extract data during the process. There is just so much data and the rate limit makes it impossible
    tiers_with_ranks = [
    ('EMERALD', ['I', 'II', 'III', 'IV']),
    # ('DIAMOND', ['I', 'II', 'III', 'IV']), #add these when I have time to collect more data
    # ('MASTER', ['I']),
    # ('GRANDMASTER', ['I']),
    # ('CHALLENGER', ['I'])
]
    print("begin fetch_puuids ------------------------------------------------------------")

    accountList = []
    for tier, ranks in tiers_with_ranks: #need to paginate as these only get first 200 accounts from each rank. Ask professor how to know how much data I need for this project? What is a significant amount of data
        for rank in ranks:
            page = 1
            while True:
                url = f'https://na1.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{tier}/{rank}?page={page}&api_key={API_TOKEN}'
                try:
                    accountPage = make_request(url)
                    if not accountPage:
                        break
                    accountList.extend(accountPage)
                    if len(accountPage) < 100:
                        break
                    page += 1
                    print(f'tier: {tier} rank: {rank} page: {page}')
                except requests.exceptions.RequestException as e:
                    print(f'Failed to get matches for {tier}, {rank}: {e}')
                    break

                time.sleep(timeouttime)
    print("end fetch_puuids ------------------------------------------------------------")
    return accountList


def fetch_matches_to_search():
    puuids = [account.puuid for account in db.session.query(Account.puuid).all()]
    print("begin fetch_matches_to_search")
    matches_to_search = set()
    for puuid in puuids:
        url = f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&type=ranked&start=0&count=2&api_key={API_TOKEN}'     #queue id 420 can change the count=20 to get more games per id.

        try:
            match_id_page = make_request(url)
            matches_to_search.update(match_id_page)
        except requests.exceptions.RequestException as e:
            print(f'Failed to get matches for {puuid}: {e}')
            continue
        time.sleep(timeouttime)
    print("end fetch_matches_to_search ------------------------------------------------------------")
    return matches_to_search

def preprocess_data(df):
    # Drop rows where any of the key fields are NaN
    df = df.dropna(subset=['puuid', 'tier','rank'])
    #change rank to numeric
    df['rank'] = df['rank'].replace({'I': 0, 'II': 100, 'III': 200, 'IV': 300}).astype(int)
    #change tier to numeric
    df['tier'] = df['tier'].str.lower().replace({'iron': 0, 'bronze': 400, 'silver': 800, 'gold': 1200, 'platinum': 1600, 'emerald': 2000, 'diamond': 2400}).astype(int)
    # Drop any rows that still have NaN values at this point (forcefully)
    df = df.dropna()
    return df

# Global variables for model
model = None

@app.route('/reload', methods=['POST']) #add another route that simply loads data from a json
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
    db.session.query(Match).delete()

    accountList = fetch_puuids()

    accountList = pd.DataFrame(accountList)
    accountList = accountList[['puuid', 'tier','rank']].dropna()
    accountList = accountList.drop_duplicates(subset="puuid", keep="first")

    for _, row in accountList.iterrows():
        new_account = Account(
            puuid=row['puuid'],
            tier=row['tier'],
            rank=row['rank'],
        )
        db.session.add(new_account)
    db.session.commit()

    matches_to_search = fetch_matches_to_search() #this is a set


    # try:
    #     with open('partial_matches.json', 'r') as f:
    #         matchesData = json.load(f)
    #         already_fetched_ids = {match['matchId'] for match in matchesData}
    # except FileNotFoundError:
    #     matchesData = []
    #     already_fetched_ids = set()

    # for matchId in matches_to_search:
    #     if matchId in already_fetched_ids:
    #         continue

    matchesData = fetch_matches(matches_to_search) #this is used if loading all data from the start

    matches_df = pd.DataFrame(matchesData)
    df_expanded = pd.DataFrame({
        'matchId': matches_df['matchId'],
        'team1Win': matches_df['team1Win'],
        'champ1': matches_df['team1Champions'].apply(lambda x: x[0] if len(x) > 0 else None),
        'champ2': matches_df['team1Champions'].apply(lambda x: x[1] if len(x) > 1 else None),
        'champ3': matches_df['team1Champions'].apply(lambda x: x[2] if len(x) > 2 else None),
        'champ4': matches_df['team1Champions'].apply(lambda x: x[3] if len(x) > 3 else None),
        'champ5': matches_df['team1Champions'].apply(lambda x: x[4] if len(x) > 4 else None),
        'champ6': matches_df['team2Champions'].apply(lambda x: x[0] if len(x) > 0 else None),
        'champ7': matches_df['team2Champions'].apply(lambda x: x[1] if len(x) > 1 else None),
        'champ8': matches_df['team2Champions'].apply(lambda x: x[2] if len(x) > 2 else None),
        'champ9': matches_df['team2Champions'].apply(lambda x: x[3] if len(x) > 3 else None),
        'champ10': matches_df['team2Champions'].apply(lambda x: x[4] if len(x) > 4 else None)
    })

    for _, row in df_expanded.iterrows():
        new_match = Match(
            matchId=row['matchId'],
            team1Win=row['team1Win'],
            champ1=row['champ1'],
            champ2=row['champ2'],
            champ3=row['champ3'],
            champ4=row['champ4'],
            champ5=row['champ5'],
            champ6=row['champ6'],
            champ7=row['champ7'],
            champ8=row['champ8'],
            champ9=row['champ9'],
            champ10=row['champ10']
        )
        db.session.add(new_match)
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
