from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import urllib
import os


#SQLALCHEMY_DATABASE_URL = "sqlite:///./banknote.db"

host_server = os.environ.get('host_server', 'noteprediction.postgres.database.azure.com')
db_server_port = urllib.parse.quote_plus(str(os.environ.get('db_server_port', '5432')))
database_name = os.environ.get('database_name', 'fastapi')
db_username = urllib.parse.quote_plus(str(os.environ.get('db_username', 'noteprediction@noteprediction')))
db_password = urllib.parse.quote_plus(str(os.environ.get('db_password', 'note@123')))
ssl_mode = urllib.parse.quote_plus(str(os.environ.get('ssl_mode','require')))
SQLALCHEMY_DATABASE_URL = 'postgresql://{}:{}@{}:{}/{}?sslmode={}'.format(db_username, db_password, host_server, db_server_port, database_name, ssl_mode)


"""engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)"""

engine = create_engine(
    #DATABASE_URL, connect_args={"check_same_thread": False}
    SQLALCHEMY_DATABASE_URL, pool_size=3, max_overflow=0
)


SessionLocal= sessionmaker(autoflush=False,autocommit=False,bind=engine)

Base = declarative_base()
