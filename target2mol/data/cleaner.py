import json
from tqdm import tqdm 


def extract_drug_target_activity_data():
    '''
    Note i opted to iterate over each file seperatly 
    as opposed to load all files into memory and iterate 
    once as memory is the limiting factor not compute
    ends up with O(4n) instead of O(n)
    '''
    # Get all targets
    print('Extracting Targets')
    with open('data/database/targets.txt', 'r') as f:
        target_data = f.read()
    target_data = target_data.split("\t\n\t\n")[1:]
    targets, target_names, target_structures = [], [], []
    for target in tqdm(target_data):
        if target[0] == '>':
            targets.append(target[1:7].strip(" ").strip("\n"))
            target_names.append(" ".join(target.split("\n")[0].split(" ")[1:]))
            target_structures.append("".join(target.split("\n")[1:]).replace('\t', ''))

    print('Filtering Targets...')
    filtered_targets, drugs, activities = [], [], []
    with open('data/database/activity.txt', 'r') as f:
        activity = f.readlines()[1:]
        for idx, line in enumerate(tqdm(activity)):
            target = line[0:6].strip(" ").strip("\n")
            if target in targets: 
                target_structure = target_structures[targets.index(target)]
                target_name = target_names[targets.index(target)]
                filtered_targets.append({'id':target, 
                                         'name':target_name,
                                         'structure':target_structure})
                drugs.append(line.split("\t")[1])
                activities.append(line.split("\t")[-1].strip("\n"))

    print('Filtering out drugs with data...')
    data = []
    with open('data/database/drugs.txt', 'r') as f:
        drug_data = f.read().split("\t\n")
    for line in tqdm(drug_data):
        if line[:6] in drugs:
            idx = drugs.index(line[:6])
            inchi = line.split("DRUINCHI")[-1].strip('\t\n')
            data.append(
                {
                'target':filtered_targets[idx],
                'drug':{'id':drugs[idx], 'inchi':inchi},
                'activity':activities[idx]
                })            
    with open('data/database/processed/data.json', 'w') as f:
        json.dump(data, f)
        
    print(f'Done, a total of {len(data)} was found, check data/database/processed/data.json')