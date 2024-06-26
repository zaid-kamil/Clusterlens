from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from datetime import datetime
from sqlalchemy.orm import sessionmaker

# base model class
Base = declarative_base()

    
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    email = Column(String(64), unique=True)
    password = Column(String(64))
    created_at = Column(DateTime, default=datetime.now)

    def __str__(self):
        return self.username
    
class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    path = Column(String(255))
    user_id = Column(Integer, ForeignKey('users.id'))
    description = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)

    def __str__(self):
        return self.name


# utility functions
def open_db():
    engine = create_engine('sqlite:///project.db', echo=True)
    session = sessionmaker(bind=engine)
    return session()


def add_to_db(object):
    db = open_db()
    db.add(object)
    db.commit()
    db.close()


if __name__ == "__main__":
    # create engine
    # mysql settings
    # database_name = 'projectdb'
    # host = 'localhost'
    # username = 'root'
    # password = 'root'
    # port = 3306
    # engine = engine.create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}')
    # demo
    engine = create_engine('sqlite:///project.db', echo=True)
    Base.metadata.create_all(engine)