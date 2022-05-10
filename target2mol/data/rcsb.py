import json
import os
from tqdm import tqdm
from pypdb import Query
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from nltk.metrics.distance import edit_distance
from pypdb.clients.pdb.pdb_client import get_pdb_file


def filter_protein_seq(protein_txt, smiles_txt, mismatch_threshold):
    '''
    Given a protein sequence checks if the PDB file
    exists for a protein that matches to within the 
    mismatch threshold 
    protein_txt should be a file with one protein seq per line
    each amino acid letter seperated by a space character
    '''
    def _save(pdb_id, smiles, distance, data):
        if pdb_id is None: return data
        print(f'Saving, {pdb_id}')
        data.append({
                'smiles':smiles,
                'protein_id': pdb_id,
                'distance_from_org':distance,
            })
        if os.path.exists(f'target2mol/data/database/proteins/{pdb_id}.pdb'):
            return data
        else:
            with open(f'target2mol/data/database/proteins/{pdb_id}.pdb', 'w') as f:
                f.write(pdb_file)
            return data
    
    with open(protein_txt, 'r') as f:
        proteins = f.readlines()
    with open(smiles_txt, 'r') as f:
        smiles = f.readlines()
    data = []
    prev_seq = ''
    for i, seq in enumerate(tqdm(proteins)):
        seq = "".join(seq.split(" ")).strip("\n")
        if prev_seq == seq and seq is not None:
            data = _save(pdb_id, smiles[i], distance, data)
            continue
        
        pdbs = Query(seq, query_type='sequence', return_type='polymer_entity').search()
        if pdbs is None: continue
        # Save the first most similar sequence
        matching_seq = pdbs['result_set'][0]['services'][0]['nodes'][0]['match_context'][0]['query_aligned_seq']
        pdb_id = pdbs['result_set'][0]['identifier'].split("_")[0]
        distance = edit_distance(seq, matching_seq,
                                substitution_cost=1,
                                transpositions=False)
        print(f'Matching Sequence is {distance} different to original')
        if distance > mismatch_threshold:
            continue
        else:
            pdb_file = get_pdb_file(pdb_id)
            if pdb_file is not None:
                data = _save(pdb_id, smiles[i], distance, data)
        prev_seq = seq
        if i % 10 == 0: 
            
            with open('target2mol/data/database/data.json', 'r') as f:
                json_data = json.loads(f.read())
            new_data = json_data + data
            with open('target2mol/data/database/data.json', 'w') as f:
                f.write(json.dumps(new_data))

            data = []

