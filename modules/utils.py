import random
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from collections import deque, namedtuple

class FeatureExtractor:
    def __init__(self, fingerprint_len, fingerprint_radius):
        self.fingerprint_len = fingerprint_len
        self.fingerprint_radius = fingerprint_radius
        
    def compute_features(self, smiles):
        '''
        Calculates morgan fingerprint for each smiles string in list
        param: smiles: list of smiles strings
        param: fingerprint_len: length of fingerprint
        '''
        features = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fingerprint_radius, self.fingerprint_len)
            output = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, output)
            features.append(output)
            
        return np.array(features)


Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self):
        self.memory = deque([])
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(
            state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*batch))
        return batch
    
    @property
    def __len__(self):
        return len(self.memory)