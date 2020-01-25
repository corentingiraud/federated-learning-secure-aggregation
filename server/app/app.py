import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from timeMetrics import timeMetrics
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sslib import shamir
from threading import Timer
import numpy as np
import sys
import json
import models
from modelTf import ModelTf
import atexit
import base64
import datetime
import pickle

# Build simple model
MODEL = ModelTf()

# CONST

TIMEOUT_SECONDS = 2
THRESHOLD = 21
CLIENT_SECRET_SIZE = len(MODEL.toNumpyFlatArray())
BYTES_NUMBER = 4
# 1: Training on server
# 2: Training on device + non-secure agregation
# 3: Training on device + secure agregation
MODE=3

app = Flask(__name__)

mysql_engine = create_engine('postgresql://postgres:root@db/federated') # , echo=True)
models.models.Base.metadata.create_all(mysql_engine)
session_factory = sessionmaker(bind=mysql_engine)
Session = scoped_session(session_factory)
Metrics = timeMetrics()

@app.route("/")
def index():
    return jsonify(
        status='Running',
        message='AndroidFederatedServer is running',
    )

@app.route("/model")
def getModel():
    modelNumpy = MODEL.toNumpyFlatArray()
    return base64.b64encode(modelNumpy)

@app.route("/tries/current")
def currentTry():
    session = Session()
    
    tryEntity = session.query(models.models.TryEntity)\
        .filter_by(currentRound=0)\
        .order_by(models.models.TryEntity.id.desc())\
        .first()

    id = None

    if tryEntity == None:
        firstTry = models.models.TryEntity()
        firstTry.currentRound = 0
        firstTry.timeoutSeconds = TIMEOUT_SECONDS
        firstTry.threshold = THRESHOLD
        session.add(firstTry)
        session.commit()
        id = firstTry.id

        #timer Start
        Metrics.addTime("Start")

    else:
        id = tryEntity.id
    
    session.close()

    return jsonify(
        idTry=id,
    )

@app.route("/tries/<idTry>/initial-params", methods=['GET'])
def initialParams(idTry):
    session = Session()

    tryEntity = session.query(models.models.TryEntity).get(int(idTry))

    if tryEntity == None or tryEntity.currentRound != 0:
        session.close()
        return 'No try with this id or try is closed', 400

    client = models.models.Client()
    client.tryEntities.append(tryEntity)
    
    # Init empty list
    defaultShamirSharesList = {
        'required_shares': THRESHOLD - 1,
        'prime_mod': None,
        'shares': []
    }

    client.suSKSharesBase64 = str(base64.b64encode(json.dumps(defaultShamirSharesList).encode('utf8')), 'utf8')
    client.buSharesBase64 = str(base64.b64encode(json.dumps(defaultShamirSharesList).encode('utf8')), 'utf8')

    session.add(client)
    session.commit()
    
    return jsonify(
        idUser=client.id,
        threshold=tryEntity.threshold
    )

@app.route("/tries/<idTry>/rounds/0/public-keys", methods=['POST'])
def postPublicKeys(idTry):
    session = Session()

    userId = request.args.get('userId')

    currentClient = session.query(models.models.Client).with_for_update(read=True, nowait=True).get(int(userId))
    currentTry = session.query(models.models.TryEntity)\
        .join(models.models.TryEntity.clients)\
        .filter(models.models.TryEntity.id==int(idTry))\
        .first()

    if currentTry.currentRound != 0:
        session.close()
        return "Try is not in round 0", 400

    if currentClient == None:
        session.close()
        return "Client not found", 400

    currentClient.publicKeyCuPEM = request.json.get('CuPK')
    currentClient.publicKeySuPEM = request.json.get('SuPK')

    readyClients = []

    for client in currentTry.clients:
        if client.publicKeyCuPEM and client.publicKeySuPEM:
            readyClients.append(client)

    if len(readyClients) >= currentTry.threshold and currentTry.thresholdReachDate == None:
        now = datetime.datetime.now()
        currentTry.thresholdReachDate = now + datetime.timedelta(seconds=currentTry.timeoutSeconds)
    
    session.commit()
    
    return json.dumps(currentClient.toRound0())

@app.route("/tries/<idTry>/rounds/1/public-keys")
def getPublicKeys(idTry):
    session = Session()

    tryEntity = session.query(models.models.TryEntity)\
        .with_for_update()\
        .join(models.models.TryEntity.clients)\
        .filter(models.models.TryEntity.id==int(idTry))\
        .first()
    
    if tryEntity == None:
        session.close()
        return 'No try with this id', 400
    
    if tryEntity.currentRound == 0:
        if tryEntity.thresholdReachDate and datetime.datetime.now() >= tryEntity.thresholdReachDate:
            tryEntity.thresholdReachDate = None
            tryEntity.currentRound = 1
            #timer round 1
            Metrics.addTime("Round1")

        else:
            session.close()
            return 'Not enough clients yet. Try again later.', 404

    elif tryEntity.currentRound != 1:
        session.close()
        return 'Try is not in round 1', 400

    readyClients = []

    for client in tryEntity.clients:
        if client.publicKeyCuPEM and client.publicKeySuPEM:
            readyClients.append(client)

    session.commit()
    
    return json.dumps(list(map(lambda client: client.toRound1(), readyClients)))

@app.route("/tries/<idTry>/rounds/1/ciphertexts", methods=['POST'])
def postCiphertext(idTry):
    userId = request.args.get('userId')

    session = Session()

    currentClient = session.query(models.models.Client).get(int(userId))
    currentTry = session.query(models.models.TryEntity)\
        .join(models.models.TryEntity.clients)\
        .filter(models.models.TryEntity.id==int(idTry))\
        .with_for_update(of=models.models.Client)\
        .first()

    if currentTry.currentRound != 1:
        return "Try is not in round 1", 400

    if currentClient == None:
        return "Client not found", 400

    currentClient.ciphertextsBase64 = str(base64.b64encode(json.dumps(request.json['ciphertexts']).encode('utf8')), 'utf8')

    suSKShares = json.loads(str(base64.b64decode(currentClient.suSKSharesBase64.encode('utf8')), 'utf8'))
    suSKShares['prime_mod'] = request.json['suSKPrimeMod']
    currentClient.suSKSharesBase64 = str(base64.b64encode(json.dumps(suSKShares).encode('utf8')), 'utf8')
    
    buShares = json.loads(str(base64.b64decode(currentClient.buSharesBase64.encode('utf8')), 'utf8'))
    buShares['prime_mod'] = request.json['buPrimeMod']
    currentClient.buSharesBase64 = str(base64.b64encode(json.dumps(buShares).encode('utf8')), 'utf8')

    readyClients = []

    for client in currentTry.clients:
        if client.ciphertextsBase64:
            readyClients.append(client)
    
    if len(readyClients) >= currentTry.threshold and currentTry.thresholdReachDate == None:
        now = datetime.datetime.now()
        currentTry.thresholdReachDate = now + datetime.timedelta(seconds=currentTry.timeoutSeconds)

    session.commit()

    return '', 200, {'ContentType':'application/json'}

@app.route("/tries/<idTry>/rounds/2/ciphertexts")
def getCiphertexts(idTry):
    session = Session()

    userId = request.args.get('userId')

    currentClient = session.query(models.models.Client).get(int(userId))

    tryEntity = session.query(models.models.TryEntity)\
        .join(models.models.TryEntity.clients)\
        .filter(models.models.TryEntity.id==int(idTry))\
        .first()
    
    if tryEntity == None:
        session.close()
        return 'No try with this id', 400
    
    if tryEntity.currentRound == 1:
        if tryEntity.thresholdReachDate and datetime.datetime.now() >= tryEntity.thresholdReachDate:
            tryEntity.thresholdReachDate = None
            tryEntity.currentRound = 2
            #timer round 2
            Metrics.addTime("Round2")
        else:
            session.close()
            return 'Not enough clients yet. Try again later.', 404

    if tryEntity.currentRound != 2:
        session.close()
        return 'Try is not in round 2', 400

    if currentClient == None:
        return "Client not found", 400

    ciphertextsList = []

    for client in tryEntity.clients:
        if client.ciphertextsBase64:
            currentsCiphertexts = json.loads(base64.decodestring(client.ciphertextsBase64.encode('utf8')))
            for ciphertextObj in currentsCiphertexts:
                if ciphertextObj['id'] == int(userId):
                    res = {
                        'id': client.id,
                        'ciphertext': ciphertextObj['ciphertext'],
                        'publicKeySu': client.publicKeySuPEM,
                        'publicKeyCu': client.publicKeyCuPEM,
                    }
                    ciphertextsList.append(res)

    session.commit()
    
    return json.dumps(ciphertextsList), 200, {'ContentType':'application/json'}

@app.route("/tries/<idTry>/rounds/2/masked-vector", methods=['POST'])
def postMaskedInputVectors(idTry):
    session = Session()

    userId = request.args.get('userId')

    currentClient = session.query(models.models.Client).get(int(userId))
    currentTry = session.query(models.models.TryEntity)\
        .with_for_update(of=models.models.Client)\
        .join(models.models.TryEntity.clients)\
        .filter(models.models.TryEntity.id==int(idTry))\
        .first()

    if currentTry == None:
        session.close()
        return 'No try with this id', 400

    if currentTry.currentRound != 2:
        session.close()
        return 'Try is not in round 2', 400

    if currentClient == None:
        return "Client not found", 400

    # request.data (=maskedVector) is already in base64 format
    currentClient.maskedVectorBase64 = str(request.data, 'utf8')

    readyClients = []

    for client in currentTry.clients:
        if client.maskedVectorBase64:
            readyClients.append(client)
    
    if len(readyClients) >= currentTry.threshold and currentTry.thresholdReachDate == None:
        now = datetime.datetime.now()
        currentTry.thresholdReachDate = now + datetime.timedelta(seconds=currentTry.timeoutSeconds)

    session.commit()

    return '', 200, {'ContentType':'application/json'}

@app.route("/tries/<idTry>/rounds/4/user-list")
def getUserList(idTry):
    session = Session()

    tryEntity = session.query(models.models.TryEntity)\
        .join(models.models.TryEntity.clients)\
        .filter(models.models.TryEntity.id==int(idTry))\
        .first()
    
    if tryEntity == None:
        session.close()
        return 'No try with this id', 400
    
    if tryEntity.currentRound == 2:
        if tryEntity.thresholdReachDate and datetime.datetime.now() >= tryEntity.thresholdReachDate:
            tryEntity.thresholdReachDate = None
            tryEntity.currentRound = 4
            #timer round 4
            Metrics.addTime("Round4")
        else:
            session.close()
            return 'Not enough clients yet. Try again later.', 404

    if tryEntity.currentRound != 4:
        session.close()
        return 'Try is not in round 4', 400

    clientList = []

    for client in tryEntity.clients:
        if client.maskedVectorBase64:
            clientList.append(client.toRound4())

    session.commit()
    
    return json.dumps(clientList), 200, {'ContentType':'application/json'}

@app.route("/tries/<idTry>/rounds/4/shares", methods=['POST'])
def postShares(idTry):
    session = Session()

    userId = request.args.get('userId')

    currentClient = session.query(models.models.Client).get(int(userId))
    currentTry = session.query(models.models.TryEntity)\
        .with_for_update(of=models.models.Client)\
        .join(models.models.TryEntity.clients)\
        .filter(models.models.TryEntity.id==int(idTry))\
        .first()

    if currentTry == None:
        session.close()
        return 'No try with this id', 400

    if currentTry.currentRound != 4:
        session.close()
        return 'Try is not in round 4', 400

    if currentClient == None:
        return "Client not found", 400

    # request.data (=maskedVector) is already in base64 format
    shares = request.json

    idClients = [client['id'] for client in shares]

    for client in currentTry.clients:
        if client.id in idClients:
            share = [share for share in shares if share['id'] == client.id][0]
            if share['suSKShare']:
                suSKShares = json.loads(str(base64.b64decode(client.suSKSharesBase64.encode('utf8')), 'utf8'))
                suSKShares['shares'].append(share['suSKShare'])
                client.suSKSharesBase64 = str(base64.b64encode(json.dumps(suSKShares).encode('utf8')), 'utf8')
            elif share['buShare']:
                buShares = json.loads(str(base64.b64decode(client.buSharesBase64.encode('utf8')), 'utf8'))
                buShares['shares'].append(share['buShare'])
                client.buSharesBase64 = str(base64.b64encode(json.dumps(buShares).encode('utf8')), 'utf8')
            
    currentClient.giveShares = True

    session.commit()
    session = Session()

    currentTry = session.query(models.models.TryEntity)\
        .with_for_update(of=models.models.Client)\
        .join(models.models.TryEntity.clients)\
        .filter(models.models.TryEntity.id==int(idTry))\
        .first()

    # Check if we can compute the global sum now
    readyClients = []

    for client in currentTry.clients:
        if client.giveShares:
            readyClients.append(client)
    
    if len(readyClients) == currentTry.threshold:
        # COMPUTE GLOBAL SUM
        globalSum = np.zeros(CLIENT_SECRET_SIZE, dtype=np.dtype('d'))
        
        # AES CTR init
        ctr = modes.CTR(b'\x00' * 16)
        initialPlaintext = b'\x00' * BYTES_NUMBER * CLIENT_SECRET_SIZE
        
        for client in currentTry.clients:
            if client.maskedVectorBase64:
                r = base64.decodebytes(client.maskedVectorBase64.encode('utf8'))
                maskedVector = np.frombuffer(r, dtype=np.dtype('d'))

                # Decode buShares using shamir
                buShares = json.loads(str(base64.b64decode(client.buSharesBase64.encode('utf8')), 'utf8'))
                bu = shamir.recover_secret(shamir.from_base64(buShares))

                # Use AES CTR To compute masked vector pu
                cipherPuv = Cipher(algorithms.AES(bu), ctr, backend=default_backend())
                encryptor = cipherPuv.encryptor()

                ct = encryptor.update(initialPlaintext) + encryptor.finalize()

                pu = np.frombuffer(ct, dtype=np.dtype('i4'))
                
                globalSum = globalSum + maskedVector - pu
            
            else:
                # Import su client secret key to regenerated masked puv
                suSKShares = json.loads(str(base64.b64decode(client.suSKSharesBase64.encode('utf8')), 'utf8'))
                su = shamir.recover_secret(shamir.from_base64(suSKShares))
                
                # Parse client Su public key
                SuSK = serialization.load_pem_private_key(su, None, default_backend())
                
                for clientToRecover in currentTry.clients:
                    if clientToRecover.id != client.id:
                        publicKey = serialization.load_pem_public_key(clientToRecover.publicKeySuPEM.encode('utf-8'), default_backend())

                        # Create shared key
                        sharedKey = SuSK.exchange(ec.ECDH(), publicKey)
                        # Perform key derivation
                        derivedKey = HKDF(
                            algorithm=hashes.SHA256(),
                            length=32,
                            salt=None,
                            info=None,
                            backend=default_backend()
                        ).derive(sharedKey)
                        
                        # **** COMPUTE Puv (client shared mask)

                        cipherPuv = Cipher(algorithms.AES(derivedKey), ctr, backend=default_backend())
                        encryptor = cipherPuv.encryptor()

                        # Generate random bytes to fill Puv array
                        ct = encryptor.update(initialPlaintext) + encryptor.finalize()
                        
                        # Convert random bytes to a numpy array using unsigned BYTES_NUMBER int
                        puv = np.frombuffer(ct, dtype=np.dtype('i4'))

                        # Compute delta for current client id
                        delta = 1 if clientToRecover.id < client.id else -1
                        
                        globalSum = globalSum + delta * puv
        
        globalMean = globalSum / len(readyClients)
        
        accuracy = MODEL.test()
        app.logger.info('Server accuracy before training (with test values): ' + str(accuracy))

        MODEL.updateFromNumpyFlatArray(globalMean)
        
        accuracy = MODEL.test()
        app.logger.info('Server accuracy after training (with test values): ' + str(accuracy))

        #timer end
        Metrics.addTime("End")
        Metrics.to_csv()
        
        # Finish current try
        currentTry.currentRound = -1

    session.commit()

    return '', 200, {'ContentType':'application/json'}

############ MODE 1 ###############

@app.route("/non-federated", methods=['POST'])
def postDataNonFederated():
    session = Session()
    #codé par TOulToul
    Metrics.addTimeMode1("Start")
    
    client = models.models.Client()

    xData = request.json['trainXEncoded']
    yData = request.json['trainYEncoded']

    client.xTrain = xData
    client.yTrain = yData
 
    session.add(client)
    
    session.commit()

    session = Session()
    
    readyClients = session.query(models.models.Client)\
        .filter(models.models.Client.xTrain.isnot(None))\
        .filter(models.models.Client.yTrain.isnot(None))\
        .all()

    if len(readyClients) == THRESHOLD:
        globalXTrain = None
        globalYTrain = None
        
        for client in readyClients:
            xTrain = pickle.loads(base64.b64decode(client.xTrain.encode('utf8')))
            yTrain = pickle.loads(base64.b64decode(client.yTrain.encode('utf8')))
            
            if type(globalXTrain) == type(None):
                globalXTrain = xTrain
            else:
                globalXTrain = np.concatenate((globalXTrain, xTrain))
            if type(globalYTrain) == type(None):
                globalYTrain = yTrain
            else:
                globalYTrain = np.concatenate((globalYTrain, yTrain))
            
            client.xTrain = None
            client.yTrain = None

        accuracy = MODEL.test()
        app.logger.info('Server accuracy before training (with test values): ' + str(accuracy))

        accuracy = MODEL.trainModel(globalXTrain, globalYTrain)
        app.logger.info('Server accuracy at the end of training (with train values): ' + str(accuracy))
        
        accuracy = MODEL.test()
        app.logger.info('Server accuracy after training (with test values): ' + str(accuracy))

        Metrics.addTimeMode1("End")
        Metrics.to_csvMode1()

    session.commit()
         
    return '', 200, {'ContentType':'application/json'}

############ MODE 2 ###############

@app.route("/non-secure/trainable-vars", methods=['POST'])
def postTrainableVars():
    session = Session()
    #timer start
    Metrics.addTimeMode1("Start")

    newClient = models.models.Client()

    newClient.trainableVars = str(request.data, 'utf8')

    session.add(newClient)

    session.commit()
    
    # Check if we can compute the global sum now
    session = Session()

    readyClients = session.query(models.models.Client)\
        .filter(models.models.Client.trainableVars.isnot(None))\
        .all()
    
    if len(readyClients) >= THRESHOLD:
        # COMPUTE GLOBAL SUM
        globalSum = np.zeros(CLIENT_SECRET_SIZE, dtype=np.dtype('d'))

        for client in readyClients:
            r = base64.decodebytes(client.trainableVars.encode('utf8'))
            trainableVars = np.frombuffer(r, dtype=np.dtype('d'))

            client.trainableVars = None

            globalSum += trainableVars

        globalMean = globalSum / len(readyClients)

        session.commit()

        accuracy = MODEL.test()
        app.logger.info('Server accuracy before training (with test values): ' + str(accuracy))

        MODEL.updateFromNumpyFlatArray(globalMean)
        
        accuracy = MODEL.test()
        app.logger.info('Server accuracy after training (with test values): ' + str(accuracy))

        #timer end
        Metrics.addTimeMode1("End")
        Metrics.to_csvMode1()

    return '', 200, {'ContentType':'application/json'} 
  
@app.teardown_appcontext
def shutdown_session(exception=None):
    Session.remove()

def exit_handler():
    print('Gracefully closing the server...')

    session = Session()

    session.query(models.models.TryEntity).\
       update({"currentRound": -1})
    session.query(models.models.Client).\
       update({"xTrain": None, "yTrain": None, "trainableVars": None})
    session.commit()

atexit.register(exit_handler)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
