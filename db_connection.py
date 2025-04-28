# database_connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import Match  # import your Match model

# Create engine and session
engine = create_engine('sqlite:///C:/Users/xwill/OneDrive/Documents/GitHub/whichCompWon/instance/listings.db')
Session = sessionmaker(bind=engine)
session = Session()

def get_match_by_index(index):
    """Return the match at a given index."""
    match = session.query(Match).offset(index).limit(1).first()
    return match
