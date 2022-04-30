import sys
import yaml
import argparse
from modules.dqn import MolDQN


if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()

    # Dataset Commands
    if command == 'train':
        args, leftover_args = parser.parse_known_args() 
        with open('config.yaml', 'r') as f:
            config = yaml.load(f.read(), yaml.FullLoader)
        dqn = MolDQN(config)
        dqn.start()
        print('Training started, check tensorboard for logs')