import itertools
from rdkit import Chem
from rdkit.Chem import Descriptors


class 

class BaseMoleculeEnv:
    def __init__(self, config):
        '''
        starting_state: 
            - SMILES string to start from
        fix_starting_state: 
            - If its a pharmacophore you're attempting to optimize etc.
        '''
        assert config['reward_type'] in ['logP', 'QED'], \
            'Only reward types logP and QED currently supported'

        self.reward_type = config['reward_type']
        self.reward_discount = config['reward_discount']
        self.max_steps = config['max_steps']
        self.starting_state = config['starting_state']
        self.state = config['starting_state']
        self.fix_starting_state = config['fix_starting_state']
        self.atom_vocab = config['atom_vocab']
        self.allow_inter_ring_bond = config['allow_inter_ring_bond']
        self.allowed_ring_sizes = config['allowed_ring_sizes']
        self.step_counter = 0
        
        self.bond_vocab = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
        self.atom_valences = {atom: self.get_atom_valences([atom])[0]
                              for atom in self.atom_vocab}
        
    @property        
    def get_mol_state(self):
        return Chem.MolFromSmiles(self.state)
        
    def reset(self):
        self.state = self.starting_state
        self.step_counter = 0
        self.reward_discount = 0.9
        
    def get_reward(self, action):
        # Calculate reward
        mol = Chem.MolFromSmiles(action)
        if self.reward_type == 'logP':
            reward = Descriptors.MolLogP(mol)
        elif self.reward_type == 'QED':
            reward = Descriptors.qed(mol)
        # Discount reward
        self.reward_discount = self.reward_discount ** (self.max_steps - self.step_counter)
        reward *= self.reward_discount 
        return reward
    
    def update_state(self, new_state):
        self.state = new_state
        
    def get_atom_valences(self, atom_types):
        '''
        Returns number of bonds each atom can make 
        eg: [C, O] -> [4, 2]
        '''
        p_table = Chem.GetPeriodicTable()
        valences = [max(list(p_table.GetValenceList(atom))) 
                    for atom in atom_types]
        return valences
        
    def get_actions(self):
        '''
        Actions are set of smiles strings of the resulting
        structures if one where to take the action, 
        ie: chemically next possible future state
        '''
        actions = set()
        # List of new structures for each possible action
        free_valence_atoms = {}
        for i in range(1, max(self.atom_valences.values())):
            free_valence_atoms[i] = [
                atom.GetIdx() for atom in self.get_mol_state.GetAtoms() 
                if atom.GetNumImplicitHs() >= i
            ]
        atom_actions = self.get_actions_for_atoms(free_valence_atoms)
        bond_addition_actions = self.get_bond_addition_actions(free_valence_atoms)
        bond_removal_actions = self.get_bond_removal_actions()
        actions = atom_actions | bond_addition_actions | bond_removal_actions
        return list(actions)
    
    def take_action(self, action):
        done = False
        self.state = action
        self.step_counter += 1
        reward = self.get_reward(action)
        if self.step_counter == self.max_steps:
            print('Max steps reached, ending session')
            done = True
        return self.state, reward, done
    
    def get_actions_for_atoms(self, free_valence_atoms):
        '''
        param: atom_valences: dict mapping atom type (str) to valence (int)
        param: free_valence_atoms: dict 
        {# of free valences: [atom_idxs]}
        Trys every combination of atom and bond to every molecule,
        if valid adds to possible actions
        '''
        state = Chem.MolFromSmiles(self.state)
        atom_actions = set()
        for i in self.bond_vocab:
            for atom in free_valence_atoms[i]:
                for element in self.atom_vocab:
                    if self.atom_valences[element] >= i:
                        new_state = Chem.RWMol(state)
                        idx = new_state.AddAtom(Chem.Atom(element))
                        new_state.AddBond(atom, idx, self.bond_vocab[i])
                        if Chem.SanitizeMol(new_state, catchErrors=True):
                            continue
                        else:
                            atom_actions.add(Chem.MolToSmiles(new_state))
        return atom_actions

    def get_bond_addition_actions(self, free_valence_atoms):
        '''
        Adds actions that don't form too large rings
        or inter ring bonds
        '''        
        state = Chem.MolFromSmiles(self.state)
        bond_vocab = list([None, *self.bond_vocab.values()])
        bond_actions = set()
        for free_valence, atom_idxs in free_valence_atoms.items():
            for atom1, atom2 in itertools.combinations(atom_idxs, 2):
                bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
                new_state = Chem.RWMol(state)
                # Kekulize to prevent sanitation errors, original state remains unmodified
                Chem.Kekulize(new_state, clearAromaticFlags=True)
                if bond is not None:
                    if bond.GetBondType() not in bond_vocab:
                        continue # Skips aromatic bonds
                    idx = bond.GetIdx()
                    bond_type = bond_vocab.index(bond.GetBondType())
                    bond_type += free_valence
                    if bond_type < len(bond_vocab):
                        idx = bond.GetIdx()
                        bond.SetBondType(bond_vocab[bond_type])
                        new_state.ReplaceBond(idx, bond)
                    else: continue
                # Stop inter ring bonding 
                elif not self.allow_inter_ring_bond and (
                    new_state.GetAtomWithIdx(atom1).IsInRing() and
                    new_state.GetAtomWithIdx(atom2).IsInRing()
                ): continue
                # Prevent large rings being formed
                elif (self.allowed_ring_sizes is not None and
                      len(Chem.rdmolops.GetShortestPath(state, atom1, atom2))
                      not in self.allowed_ring_sizes
                      ): continue
                else:
                    new_state.AddBond(atom1, atom2, bond_vocab[free_valence])
                if Chem.SanitizeMol(new_state, catchErrors=True):
                    continue
                bond_actions.add(Chem.MolToSmiles(new_state))
                
        return bond_actions
                    
    def get_bond_removal_actions(self):
        '''
        Valid actions are:
        triple -> double, single, none
        double -> single, none
        single -> none
        Aromatic bonds are not removed
        More than one molecular fragment is not allowed
        '''
        if len(self.state) <= 2: return set() 
        state = Chem.MolFromSmiles(self.state)
        bond_vocab = [None, *self.bond_vocab.values()]
        bond_actions = set()
        for valence in [1,2,3]:
            for bond in state.GetBonds():
                bond = Chem.Mol(state).GetBondBetweenAtoms(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                if bond.GetBondType() not in bond_vocab:
                    continue # Skips aromatic
                new_state = Chem.RWMol(state)
                # Kekulize to avoid sanitation errors
                Chem.Kekulize(new_state, clearAromaticFlags=True)
                bond_value = bond_vocab.index(bond.GetBondType())
                bond_value -= valence
                if bond_value > 0: # Remove bond
                    atom1 = bond.GetBeginAtom().GetIdx()
                    atom2 = bond.GetEndAtom().GetIdx()
                    new_state.RemoveBond(atom1, atom2)
                    if Chem.SanitizeMol(new_state, catchErrors=True):
                        continue # Invalid bond removal
                    smiles = Chem.MolToSmiles(new_state)
                    # Check only one molecular fragment exists
                    parts = sorted(smiles.split("."), key=len)
                    if len(parts) == 1 or len(parts[0]) == 1:
                        bond_actions.add(parts[-1])
        return bond_actions