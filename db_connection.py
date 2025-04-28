# database_connection.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import Match  # import your Match model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'instance', 'listings.db')

# Create engine and session
engine = create_engine(f'sqlite:///{db_path}')
Session = sessionmaker(bind=engine)
session = Session()

def get_match_by_index(index):
    """Return the match at a given index."""
    match = session.query(Match).offset(index).limit(1).first()
    return match
