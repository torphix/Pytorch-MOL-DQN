https://pypi.org/project/proteingraph/

A model that is trained on target & molecule data
That at inference time takes as input a pdb file and outputs a series of suggested molecules


Databases:
    - http://biomine.cs.vcu.edu/servers/PDID/index.php#Database
    - https://www.bindingdb.org/bind/info.jsp



Inference:
    Protein strucuture -> Embedding -> Graph representation -> Graph embedding -> Neural network -> Candidate molecules
Train:
    Molecular structure -> Embedding -> Graph Repres


How to generate a molecular graph from latent protein vector space
    Option 1: 
        - Have MLP predict Adjacency matrix (bond types) and annotation matrix (atom types)
