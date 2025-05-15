CREATE SCHEMA IF NOT EXISTS league;

DROP TABLE IF EXISTS league.accounts CASCADE;

CREATE TABLE IF NOT EXISTS league.accounts(
    puuid                         TEXT PRIMARY KEY
,   tier                          TEXT
,   rank                          TEXT
);

DROP TABLE IF EXISTS league.searched_matches CASCADE;

CREATE TABLE IF NOT EXISTS league.searched_matches(
    puuid                         TEXT PRIMARY KEY
);

DROP TABLE IF EXISTS league.matches CASCADE;

CREATE TABLE IF NOT EXISTS league.matches(
    match_id                 TEXT PRIMARY KEY
,   team1win                 BOOLEAN
,   champ1                   TEXT
,   champ2                   TEXT
,   champ3                   TEXT
,   champ4                   TEXT
,   champ5                   TEXT
,   champ6                   TEXT
,   champ7                   TEXT
,   champ8                   TEXT
,   champ9                   TEXT
,   champ10                  TEXT
);