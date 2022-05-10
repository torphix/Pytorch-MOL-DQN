# Finish molDQN
# 
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem

epinephrine = Chem.MolFromSmiles('CNC[C@H](O)c1ccc(O)c(O)c1')
fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(epinephrine, radius=3)
Draw.DrawMorganBit(epinephrine,589,bi)
