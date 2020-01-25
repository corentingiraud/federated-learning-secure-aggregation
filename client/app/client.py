import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import requests as rq
import json
import secrets
import time
import base64

from random import randrange
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sslib import shamir, randomness
import numpy as np
import pickle
from modelTf import ModelTf

# Build simple model
MODEL = ModelTf()

# CONST
SERVER_IP = 'http://localhost:3000' # os.environ['SERVER_IP'] or 
PULL_REQUEST_INTERVAL = 1 # in second(s)
HEADERS = { 'Content-type': 'application/json' }
BYTES_NUMBER = 4
# 1: Training on server
# 2: Training on device + non-secure agregation
# 3: Training on device + secure agregation
MODE=3

print('****** CLIENT BEGIN IN MODE ' + str(MODE))

#!!!!!!!!!!!!!!!!!!!!!! MODE 1
if MODE == 1:
    # Train locally the model
    randomIndex = randrange(0,5)
    print('Client will send part ' + str(randomIndex) + ' of the dataset to the server')
    trainX, trainY = MODEL.getTrainingData(5, randomIndex)

    trainXEncoded = pickle.dumps(trainX)
    trainYEncoded = pickle.dumps(trainY)

    data = {'trainXEncoded': base64.b64encode(trainXEncoded).decode('utf8'), 'trainYEncoded': base64.b64encode(trainYEncoded).decode('utf8')}
    data_json = json.dumps(data)
    res = rq.post(SERVER_IP+'/non-federated', data=data_json, headers=HEADERS)
    
    print('Client finish')
    exit(0) 

# Get current model
res = rq.get(SERVER_IP+'/model')
r = base64.b64decode(res.text)
initialTrainableVars = np.frombuffer(r, dtype=np.dtype('d'))
MODEL.updateFromNumpyFlatArray(initialTrainableVars)

#!!!!!!!!!!!!!!!!!!!!!! MODE 2
if MODE == 2:
    # Train locally the model
    randomIndex = randrange(0,5)
    print('Client will train with part ' + str(randomIndex) + ' of the dataset')
    acc = MODEL.trainModel(5, randomIndex)
    print('Client accuracy after training (with train values): ' + str(acc))

    xu = MODEL.toNumpyFlatArray().copy()
    
    xuEncoded = base64.b64encode(xu)
    res = rq.post(SERVER_IP+'/non-secure/trainable-vars', data=xuEncoded)
    
    print('Client finish')
    exit(0)

#!!!!!!!!!!!!!!!!!!!!!! MODE 3

############## MODEINIT ##############

# Get current try
res = rq.get(SERVER_IP+'/tries/current')
initialParams = res.json()
idTry = str(initialParams['idTry'])

# Get initial params
res = rq.get(SERVER_IP+'/tries/'+idTry+'/initial-params')
initialParams = res.json()
threshold = initialParams['threshold']
idUser = initialParams['idUser']
# Train locally the model
randomIndex = randrange(0,25)
print('Client ' + str(idUser) + ' will train with part ' + str(randomIndex) + ' of the dataset')

acc = MODEL.trainModel(25, randomIndex)
print('Client ' + str(idUser) + ' accuracy after training (with test values): ' + str(acc))

############## ROUND 0 ##############

# Generate two pairs of keys Cu and Su

CuSK = ec.generate_private_key(
    ec.SECP384R1(),
    default_backend())
CuPK = CuSK.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

SuSK = ec.generate_private_key(
    ec.SECP384R1(),
    default_backend())
SuPK = SuSK.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

keys = {'CuPK': str(CuPK, "utf-8"), 'SuPK': str(SuPK, "utf-8")}
keys_json = json.dumps(keys)

# Send public keys to server
res = rq.post(SERVER_IP+'/tries/'+idTry+'/rounds/0/public-keys?userId='+str(idUser), data=keys_json, headers=HEADERS)

############## ROUND 1 ##############

# Get U1 the list with public keys from all clients
url = SERVER_IP+'/tries/'+idTry+'/rounds/1/public-keys'
res = rq.get(url)

while res.status_code != 200:
    res = rq.get(url)
    time.sleep(PULL_REQUEST_INTERVAL)

clientsU1 = res.json()

# Generate random client mask vector bu
bu = secrets.token_bytes(16)

# Split bu into n (=client numbers) shares
shamirResBu = shamir.split_secret(
    bu,
    threshold - 1,
    len(clientsU1) - 1,
    randomness_source=randomness.UrandomReader()
)
# Encode shares with base64
shamirResBu = shamir.to_base64(shamirResBu)

sharesBu = shamirResBu['shares']

# Splitting SuSK into n (=client numbers) shares
SuSK_as_string = SuSK.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

shamirResSuSK = shamir.to_base64(shamir.split_secret(SuSK_as_string, threshold - 1, len(clientsU1) - 1))
sharesSuSK = shamirResSuSK['shares']

ciphertexts = []
i = 0

# Compute cipher text for each client
for client in clientsU1:
    if client['id'] != idUser:
        toBeEncrypted = str(idUser) + ';' + str(client['id']) + ';' + sharesSuSK[i] + ';' + sharesBu[i]
        # Parse client Cu public key
        publicKey = serialization.load_pem_public_key(client['publicKeyCu'].encode('utf-8'), default_backend())
        # Create shared key
        sharedKey = CuSK.exchange(ec.ECDH(), publicKey)
        # Perform key derivation and encode key
        derivedKey = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=None,
            backend=default_backend()
        ).derive(sharedKey)
        key = base64.urlsafe_b64encode(derivedKey)
        f = Fernet(key)
        ciphertexts.append({
            'id': client['id'],
            'ciphertext': str(f.encrypt(toBeEncrypted.encode('utf8')), "utf-8")
        })
        # Save fernet instance for future use (round 4)
        client['fernet'] = f
        i += 1

reqData = {
    'suSKPrimeMod': shamirResSuSK['prime_mod'],
    'buPrimeMod': shamirResBu['prime_mod'],
    'ciphertexts': ciphertexts,
}
reqJson = json.dumps(reqData)
res = rq.post(SERVER_IP+'/tries/'+idTry+'/rounds/1/ciphertexts?userId='+str(idUser), data=reqJson, headers=HEADERS)

############## ROUND 2 ##############

# Get U2 the list of ciphertexts from all clients
url = SERVER_IP + '/tries/' + idTry + '/rounds/2/ciphertexts?userId=' + str(idUser)
res = rq.get(url)

if randrange(0,100) >= 90:
    print('Client ' + str(idUser) + ': internet connection lost. EXIT.')
    exit(0)

while res.status_code != 200:
    res = rq.get(url)
    time.sleep(PULL_REQUEST_INTERVAL)

clientsU2 = res.json()

maskedVector = MODEL.toNumpyFlatArray().copy()

# Iterate through clients (U2)

# We are going to use AES in CTR mode as a pseudo random generator
# to generate Puv & Pu
# CTR is configured with full zero nounce
# AES will encrypt full zero plaintext every time but using different key
ctr = modes.CTR(b'\x00' * 16)
initialPlaintext = b'\x00' * BYTES_NUMBER * maskedVector.size

for client in clientsU2:

    if client['id'] == idUser:
        raise Exception('ROUND2: idUser schouldn\'t be equal to received client')
    # Parse client Su public key
    SuPkClient = serialization.load_pem_public_key(client['publicKeySu'].encode('utf-8'), default_backend())
    
    # Create shared key
    sharedKey = SuSK.exchange(ec.ECDH(), SuPkClient)
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

    # Convert random bytes to a numpy array
    puv = np.frombuffer(ct, dtype=np.dtype('i4'))

    # Compute delta for current client id
    delta = 1 if client['id'] < idUser else -1

    # Add Puv to the maskedVector
    maskedVector += delta * puv

# **** COMPUTE Pu (client personal mask)
cipherPu= Cipher(algorithms.AES(bu), ctr, backend=default_backend())
encryptor = cipherPu.encryptor()

# Generate random bytes to fill Pu array
ct = encryptor.update(initialPlaintext) + encryptor.finalize()

# Convert random bytes to a numpy array
pu = np.frombuffer(ct, dtype=np.dtype('i4'))

# Finally add it to the masked vector
maskedVector += pu

# Send base64 encoded maskedVector to the server
# Encode it using base64
# https://stackoverflow.com/a/6485943

maskedVectorEncoded = base64.b64encode(maskedVector)
res = rq.post(SERVER_IP+'/tries/'+idTry+'/rounds/2/masked-vector?userId='+str(idUser), data=maskedVectorEncoded)

############## ROUND 3 ##############

# As round 3 is only to garante authenticity, we choose to skip it 

############## ROUND 4 ##############

# Get U4 the list of ciphertexts from all clients
url = SERVER_IP + '/tries/' + idTry + '/rounds/4/user-list'
res = rq.get(url)

while res.status_code != 200:
    res = rq.get(url)
    time.sleep(PULL_REQUEST_INTERVAL)

clientsU4 = res.json()

resClientU4 = []

# Send either a share of SuSK or
for clientU2 in clientsU2:
    clientU1 = next((clientU1 for clientU1 in clientsU1 if clientU1["id"] == clientU2['id']), None)
    
    decryptedCipher = str(clientU1['fernet'].decrypt(clientU2['ciphertext'].encode('utf8')), 'utf8')
    decryptedCipher = decryptedCipher.split(';')

    clientU4 = next((clientU4 for clientU4 in clientsU4 if clientU4["id"] == clientU2['id']), None)
    
    if clientU4:
        resClientU4.append({
            'id': int(clientU2['id']),
            'buShare': decryptedCipher[3],
            'suSKShare': None,
        })
    else:
        resClientU4.append({
            'id': int(clientU2['id']),
            'buShare': None,
            'suSKShare': decryptedCipher[2],
        })


resClientU4Json = json.dumps(resClientU4)
res = rq.post(SERVER_IP+'/tries/'+idTry+'/rounds/4/shares?userId='+str(idUser), data=resClientU4Json, headers=HEADERS)

print('Client ' + str(idUser) + ' finished.')
