from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Text, Table, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
import sqlalchemy.sql.functions as func
##### INIT

Base = declarative_base()

##### [ASSOCIATION] Client // TryEntity

association_clients_tryEntities = Table('clients-tryEntities', Base.metadata,
    Column('idClient', Integer, ForeignKey('clients.id')),
    Column('idTryEntity', Integer, ForeignKey('tryEntities.id'))
)


##### [ENTITY] Client

class Client(Base):
    __tablename__ = 'clients'
 
    id = Column(Integer, primary_key=True)
    publicKeyCuPEM = Column(Text)
    publicKeySuPEM = Column(Text)
    ciphertextsBase64 = Column(Text)
    maskedVectorBase64 = Column(Text)
    suSKSharesBase64 = Column(Text)
    buSharesBase64 = Column(Text)
    giveShares = Column(Boolean, unique=False, default=False)
    xTrain = Column(Text)
    yTrain = Column(Text)
    trainableVars = Column(Text)
    tryEntities = relationship(
        "TryEntity",
        secondary=association_clients_tryEntities,
        back_populates="clients")
    
    def toRound0(self):
        return {
            'id': self.id,
            'publicKeyCu': self.publicKeyCuPEM,
            'publicKeySu': self.publicKeySuPEM,
        }

    def toRound1(self):
        return {
            'id': self.id,
            'publicKeyCu': self.publicKeyCuPEM,
            'publicKeySu': self.publicKeySuPEM,
        }
    
    def toRound2(self):
        return {
            'id': self.id,
            'ciphertexts': self.ciphertextBase64,
        }
    
    def toRound4(self):
        return {
            'id': self.id,
        }
    
    def __repr__(self):
        return 'Client()'

    def __str__(self):
        return 'Instance of Client'


class TryEntity(Base):
    __tablename__ = 'tryEntities'
    id = Column(Integer, primary_key=True)
    currentRound = Column(Integer)
    createdAt = Column(DateTime, server_default=func.now())
    timeoutSeconds = Column(Integer)
    thresholdReachDate = Column(DateTime)
    threshold = Column(Integer)

    clients = relationship(
        "Client",
        secondary=association_clients_tryEntities,
        back_populates="tryEntities")
