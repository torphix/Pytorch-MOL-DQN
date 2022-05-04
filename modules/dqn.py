import numpy as np
from tqdm import tqdm
from .agent import Agent
from .env import MoleculeEnv
from .utils import FeatureExtractor
from torch.utils.tensorboard import SummaryWriter



class MolDQN:
    def __init__(self, config):
        config['env']['max_steps'] = config['max_steps']
        
        self.env = MoleculeEnv(config['env'])
        self.agent = Agent(config['agent'])
        self.feature_extractor = FeatureExtractor(config['fingerprint_len'], 
                                                  config['fingerprint_radius'])
        # Run parameters
        self.max_steps = config['max_steps']
        self.update_step = config['update_step']
        self.epochs = config['epochs']
        # Feature parameters
        self.fingerprint_len = config['fingerprint_len']
        self.fingerprint_radius = config['fingerprint_radius']
        
        self.logger = SummaryWriter(config['log_path'])
        
    def remaining_steps(self, observations, current_step):
        return 
        
    def start(self):
        '''
        1. Get current state & available actions
        2. Compute features & get agents prediction
        3. Take action and log reward
        4. Once max_steps reached update agent
        5. Loop till max_epochs reached
        '''
        for step in tqdm(range(self.epochs)):
            # Actions are acutally next future states
            actions = self.env.get_actions()
            observations = self.feature_extractor.compute_features(actions)
            # Add number of steps left so network can factor it in
            remaining_step = np.array(self.max_steps - step)
            observations = np.stack([np.append(o, remaining_step) for o in observations])
            action_idx = self.agent.get_action(observations)
            # Action taken (aka next state), reward recieved, final step
            action, reward, done = self.env.take_action(action_idx)
            # Get next state
            next_actions = self.env.get_actions()
            next_observations = self.feature_extractor.compute_features(next_actions)
            next_observations = np.vstack([np.append(next_observation, remaining_step) 
                                           for next_observation in next_observations])
            action = np.append(self.feature_extractor.compute_features([action])[0], remaining_step)
            self.agent.memory.push(
                state=observations,
                action=action,
                reward=reward,
                next_state=next_observations,
                done=done)
            
            if step % self.update_step == 0:
                loss = self.agent.update_params(step)
                if loss is not None:
                    self.logger.add_scalar(f'Loss', loss, step)
                
            if done:
                print(f'Reward for episode: {reward}')
                print(f'Current reward discount: {self.agent.reward_discount}')
                print(f'Current search epsilon: {self.agent.search_epsilon}')
                final_reward=reward
                self.env.reset()
                self.logger.add_scalar("Episode reward", final_reward, step)