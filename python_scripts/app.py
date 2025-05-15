import psycopg2
import requests
from dbFunctions import *
# For rate limiting
import backoff
import time
timeouttime = 0.06
# to hide API keys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Boolean

DATABASE_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class Account(Base):
    __tablename__ = 'accounts'
    __table_args__ = {'schema': 'league'}

    puuid = Column(String, primary_key=True)
    tier = Column(String)
    rank = Column(String)

class Searched_Matches(Base):
    __tablename__ = 'searched_matches'
    __table_args__ = {'schema': 'league'}

    puuid = Column(String, primary_key=True)

class Matches(Base):
    __tablename__ = 'matches'
    __table_args__ = {'schema': 'league'}

    match_id = Column(String, primary_key=True)
    team1win = Column(Boolean)
    rank = Column(String)
    champ1 = Column(String)
    champ2 = Column(String)
    champ3 = Column(String)
    champ4 = Column(String)
    champ5 = Column(String)
    champ6 = Column(String)
    champ7 = Column(String)
    champ8 = Column(String)
    champ9 = Column(String)
    champ10 = Column(String)

load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT"))
)
cursor = conn.cursor()

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

def fetch_matches():
    matches_to_search = [match.puuid for match in session.query(Searched_Matches.puuid).all()]
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
                'match_id': matchId,
                'champ1': team1champions[0],
                'champ2': team1champions[1],
                'champ3': team1champions[2],
                'champ4': team1champions[3],
                'champ5': team1champions[4],
                'champ6': team2champions[0],
                'champ7': team2champions[1],
                'champ8': team2champions[2],
                'champ9': team2champions[3],
                'champ10': team2champions[4],
                'team1win': team1Win
            })
        except requests.exceptions.RequestException as e:
            print(f'Failed to get matches for {matchId}: {e}')
            continue
        time.sleep(timeouttime)
    print("end fetch_matches ------------------------------------------------------------")
    print("adding matches to db")
    for match in matchesData:
        executeInsertMatchData(cursor, match)
    cursor.connection.commit()
    return matchesData

#info -> endOfGameResult == GameComplete -> Participants -> 0 -> champion name

def fetch_puuids(): #add saftey to allow me to extract data during the process. There is just so much data and the rate limit makes it impossible
    tiers_with_ranks = [
    ('EMERALD', ['I']),#, 'II', 'III', 'IV']),
    # ('DIAMOND', ['I', 'II', 'III', 'IV']), #add these when I have time to collect more data
    # ('MASTER', ['I']),
    # ('GRANDMASTER', ['I']),
    ('CHALLENGER', ['I'])
]
    print("begin fetch_puuids ------------------------------------------------------------")

    accountList = []
    for tier, ranks in tiers_with_ranks:
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
    print("adding accounts to db")
    for account in accountList:
        executeInsertAccount(cursor, account)
    cursor.connection.commit()
    return accountList


def fetch_matches_to_search():
    puuids = [account.puuid for account in session.query(Account.puuid).all()]
    print("begin fetch_matches_to_search")
    matches_to_search = set()
    for puuid in puuids:
        url = f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&type=ranked&start=0&count=1&api_key={API_TOKEN}'     #queue id 420 can change the count=20 to get more games per id.

        try:
            match_id_page = make_request(url)
            matches_to_search.update(match_id_page)
        except requests.exceptions.RequestException as e:
            print(f'Failed to get matches for {puuid}: {e}')
            continue
        time.sleep(timeouttime)
    print("end fetch_matches_to_search ------------------------------------------------------------")
    print("adding matches to db")
    for match in matches_to_search:
        executeInsertSearchedMatch(cursor, match)
    cursor.connection.commit()
    return matches_to_search

# Global variables for model
# model = None

# @app.route('/reload', methods=['POST']) #add another route that simply loads data from a json
# def reload_data():
#     '''
#     Reload data from the League of Legends dataset, clear the database, load new data, and return summary stats
#     ---
#     responses:
#       200:
#         description: Summary statistics of reloaded data
#     '''
#     global model

#     db.session.query(Account).delete()
#     db.session.query(Match).delete()

#     accountList = fetch_puuids()

#     accountList = pd.DataFrame(accountList)
#     accountList = accountList[['puuid', 'tier','rank']].dropna()
#     accountList = accountList.drop_duplicates(subset="puuid", keep="first")

#     for _, row in accountList.iterrows():
#         new_account = Account(
#             puuid=row['puuid'],
#             tier=row['tier'],
#             rank=row['rank'],
#         )
#         db.session.add(new_account)
#     db.session.commit()

#     matches_to_search = fetch_matches_to_search() #this is a set

#     matchesData = fetch_matches(matches_to_search) #this is used if loading all data from the start

#     matches_df = pd.DataFrame(matchesData)
#     df_expanded = pd.DataFrame({
#         'matchId': matches_df['matchId'],
#         'team1Win': matches_df['team1Win'],
#         'champ1': matches_df['team1Champions'].apply(lambda x: x[0] if len(x) > 0 else None),
#         'champ2': matches_df['team1Champions'].apply(lambda x: x[1] if len(x) > 1 else None),
#         'champ3': matches_df['team1Champions'].apply(lambda x: x[2] if len(x) > 2 else None),
#         'champ4': matches_df['team1Champions'].apply(lambda x: x[3] if len(x) > 3 else None),
#         'champ5': matches_df['team1Champions'].apply(lambda x: x[4] if len(x) > 4 else None),
#         'champ6': matches_df['team2Champions'].apply(lambda x: x[0] if len(x) > 0 else None),
#         'champ7': matches_df['team2Champions'].apply(lambda x: x[1] if len(x) > 1 else None),
#         'champ8': matches_df['team2Champions'].apply(lambda x: x[2] if len(x) > 2 else None),
#         'champ9': matches_df['team2Champions'].apply(lambda x: x[3] if len(x) > 3 else None),
#         'champ10': matches_df['team2Champions'].apply(lambda x: x[4] if len(x) > 4 else None)
#     })

#     for _, row in df_expanded.iterrows():
#         new_match = Match(
#             matchId=row['matchId'],
#             team1Win=row['team1Win'],
#             champ1=row['champ1'],
#             champ2=row['champ2'],
#             champ3=row['champ3'],
#             champ4=row['champ4'],
#             champ5=row['champ5'],
#             champ6=row['champ6'],
#             champ7=row['champ7'],
#             champ8=row['champ8'],
#             champ9=row['champ9'],
#             champ10=row['champ10']
#         )
#         db.session.add(new_match)
#     db.session.commit()

#     return jsonify()

if __name__ == '__main__':
    fetch_puuids()
    fetch_matches_to_search()
    fetch_matches()
