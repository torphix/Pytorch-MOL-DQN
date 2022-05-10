# Molecular DQN
- Implementation of this paper https://www.nature.com/articles/s41598-019-47148-x#Bib1 in pytorch

## Summary
- Given an input state (Molecule) a neural network predicts the best action to take to modify the molecule add bond, remove bond, add atom (with some constraints)
- The molecule is then passed through a reward function that calculates a particular metric eg: quantitative estimate of drug-likeness (QED)
- The model is then trained to maximise that particular metric by choosing actions that will result in an increase in reward

## Train
- Create conda env `conda env create -f env.yaml`
- Train `python main.py train`

## Future Features
- [] Multi metric optimization eg: Optimize for lipenskis rule of 5 & predicted IC50 against a target
- [] Add learnable graph embeddings for molecular structure (right now morgan fingerprint being used)
- [] Input pharmacophore and desired metrics and create a series of candidate molecules
- [] Condition with protein graph embedding and use tanimoto factor to generate protein specific molecules
