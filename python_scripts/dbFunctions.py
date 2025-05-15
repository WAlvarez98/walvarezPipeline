insertAccount = """
    INSERT INTO league.accounts(
        puuid
    ,   tier
    ,   rank
    ) VALUES(
        %s, %s, %s
    ) ON CONFLICT (puuid) DO NOTHING;
"""

def executeInsertAccount(cursor, account):
    cursor.execute(insertAccount,(
        account.get('puuid')
    ,   account.get('tier')
    ,   account.get('rank')
    ))
    cursor.connection.commit()

insertSearchedMatch = """
    INSERT INTO league.searched_matches(
        puuid
    ) VALUES(
        %s
    ) ON CONFLICT (puuid) DO NOTHING;
"""

def executeInsertSearchedMatch(cursor, searched_match):
    cursor.execute(insertSearchedMatch,(
        searched_match,
    ))
    cursor.connection.commit()

insertMatchData = """
    INSERT INTO league.matches(
    match_id
,   team1win
,   champ1
,   champ2
,   champ3
,   champ4
,   champ5
,   champ6
,   champ7
,   champ8
,   champ9
,   champ10
    ) VALUES(
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    ) ON CONFLICT (match_id) DO NOTHING;
"""

def executeInsertMatchData(cursor, matchData):
    cursor.execute(insertMatchData,(
        matchData.get('match_id')
    ,   matchData.get('team1win')
    ,   matchData.get('champ1')
    ,   matchData.get('champ2')
    ,   matchData.get('champ3')
    ,   matchData.get('champ4')
    ,   matchData.get('champ5')
    ,   matchData.get('champ6')
    ,   matchData.get('champ7')
    ,   matchData.get('champ8')
    ,   matchData.get('champ9')
    ,   matchData.get('champ10')
    ))
    cursor.connection.commit()