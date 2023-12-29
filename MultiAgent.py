import torch 
from  ReplayBufferMultiAgent import ReplayBuffer
from MultiAgentNetwork import Qnetwork , SimpleNetworkWithDiffrentOptimizer , MoreLayerDiffrentLossFunction , MoreLayersNetwork , MoreLayersNetworkDiffrentOptimizer
import random 
import os 


class Agent:
 
    def __init__(self, input_dimlsit, fc1_dimlsit, fc2_dimlist, fc3_dimlist, fc4_dimlist, n_actions, lrlist, losslist,
                    batch_size, mem_size, gamma_list, num_agents):
            """
            Initializes the Agent with hyperparameters.

            Args:
            - input_dimlsit (list): List of input dimensions for each agent.
            - fc1_dimlsit (list): List of fc1 dimensions for each agent.
            - fc2_dimlist (list): List of fc2 dimensions for each agent.
            - fc3_dimlist (list): List of fc3 dimensions for each agent.
            - fc4_dimlist (list): List of fc4 dimensions for each agent.
            - n_actions (int): Number of possible actions.
            - lrlist (list): List of learning rates for each agent.
            - losslist (list): List of loss functions for each agent.
            - batch_size (list): List of batch sizes for each agent.
            - mem_size (list): List of memory sizes for each agent.
            - gamma_list (list): List of gamma values for each agent.
            - num_agents (int): Total number of agents.
            """
            self.num_agents = num_agents
            self.evaluate = False
            self.agents = []

            Networks_list = [Qnetwork, MoreLayerDiffrentLossFunction, SimpleNetworkWithDiffrentOptimizer,
                            MoreLayerDiffrentLossFunction, MoreLayersNetwork, MoreLayersNetworkDiffrentOptimizer]
            self.gamma_list = gamma_list

            for index in range(num_agents):
                input_dim = input_dimlsit[index]
                fc1_dim = fc1_dimlsit[index]
                fc2_dim = fc2_dimlist[index]
                fc3_dim = fc3_dimlist[index]
                fc4_dim = fc4_dimlist[index]
                lr = lrlist[index]
                loss = losslist[index]
                memory_size = mem_size[index]

                agent_mem = ReplayBuffer(memory_size, input_dim, n_actions)
                agent_network = Qnetwork(input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_actions, lr, loss)
                gamma = gamma_list[index]

                agent = {
                    'mem': agent_mem,
                    'network': agent_network,
                    'epsilon': 0,
                    'n_games': 0,
                    'gamma': gamma
                }

                self.agents.append(agent)
                self.batch_size = batch_size


    def choose_action(self, states):


        actions = []


        for agent_index, agent in enumerate(self.agents):


            epsilon = 1000 - agent['n_games']
            final_move = [0, 0, 0]
            
            if random.randint(0, 1000) < epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1

            else:

                state = states[agent_index]
                state_tensor = torch.tensor(state, dtype=torch.float)
                prediction = agent['network'](state_tensor)            
                move = torch.argmax(prediction).item()
                final_move[move] = 1
            actions.append(final_move)

        return actions


    def short_mem(self, states, next_states, actions, rewards, dones):
        """
        Store short-term memory transitions for each agent.

        Args:
        - states (list): List of current states.
        - next_states (list): List of next states.
        - actions (list): List of actions.
        - rewards (list): List of rewards.
        - dones (list): List of done flags.

        Returns:
        None
        """
        for agent_index, agent in enumerate(self.agents):
            agent['mem'].store_transition(states[agent_index], next_states[agent_index],
                                        actions[agent_index], rewards[agent_index], dones[agent_index])
            agent['n_games'] += 1
            agent['epsilon'] = 100 - agent['n_games']

        self.learn()


    def long_mem(self):
        """
        Execute long-term memory learning for each agent if conditions are met.

        Args:
        None

        Returns:
        None
        """
        for index, agent in enumerate(self.agents):
            if self.batch_size[index] < agent['mem'].mem_cntr:
                self.learn()

    def long_memory(self, agent_index):
        """
        Execute long-term memory learning for a specific agent if conditions are met.

        Args:
        agent_index (int): Index of the agent to perform long-term memory learning.

        Returns:
        None
        """
        agent = self.agents[agent_index]
        if self.batch_size[agent_index] < agent['mem'].mem_cntr:
            self.learn()

    def save(self, agent_idx, color, zeitpunkt):
        """
        Save the model weights of a specific agent at a given time.

        Args:
        agent_idx (int): Index of the agent to save.
        color (str): Color of the snake associated with the agent.
        zeitpunkt (str): Timestamp indicating when the model is saved.

        Returns:
        None
        """
        file_name = f'Agent{agent_idx}Color{color}TakenAt{zeitpunkt}.pth'
        model_folder_path = f'./SavedModells/{agent_idx}'
        
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        agent = self.agents[agent_idx]
        file_name_agent = os.path.join(model_folder_path, f'{file_name}_agent_{agent_idx}')
        torch.save(agent['network'].state_dict(), file_name_agent)


    def load_model(self, saved_model_path):
        """
        Load a pre-trained model from the specified path.

        Args:
        saved_model_path (str): Path to the saved model file.

        Returns:
        Qnetwork: Loaded Q-network model.
        """
        # Step 2: Load the saved model parameters
        saved_model = torch.load(saved_model_path)
        model = Qnetwork(self.input_dim, self.fc1_dim, self.fc2_dim, self.fc3_dim, self.fc4_dim, self.n_actions, self.lr, self.loss)
        model.load_state_dict(saved_model)

        return model

    
    def loadmodel (self  , saved_path_list  ) : 
         model_list  = ()
         for index  in range (self.num_agents) : 
              model  = self.load_model(saved_path_list[index])
              model_list.append(model)
         return model_list

    def learn(self):
        """
        Perform a learning step for each agent using experiences sampled from their memory.

        This function implements the Deep Q-learning algorithm.

        Returns:
        None
        """
        for agent_index, agent in enumerate(self.agents):
            states, next_states, actions, rewards, dones = agent['mem'].sample_batch(self.batch_size[agent_index])

            state_tensor = torch.tensor(states, dtype=torch.float)
            next_state_tensor = torch.tensor(next_states, dtype=torch.float)
            action_tensor = torch.tensor(actions, dtype=torch.long)
            reward_tensor = torch.tensor(rewards, dtype=torch.float)
            done_tensor = torch.tensor(dones)

            if len(state_tensor.shape) == 1:
                state_tensor = torch.unsqueeze(state_tensor, 0)
                next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
                action_tensor = torch.unsqueeze(action_tensor, 0)
                reward_tensor = torch.unsqueeze(reward_tensor, 0)
                done_tensor = torch.unsqueeze(done_tensor, 0)
            
            pred = agent['network'](state_tensor)

            target = pred.clone()
            for idx in range(len(done_tensor)):
                Q_new = reward_tensor[idx]
                if not done_tensor[idx]:
                    Q_new = reward_tensor[idx] + agent['gamma'] * torch.max(agent['network'](next_state_tensor[idx]))

                target[idx][torch.argmax(action_tensor[idx]).item()] = Q_new

            agent['network'].optimizer.zero_grad()
            loss = agent['network'].loss(target, pred)
            loss.backward()
            agent['network'].optimizer.step()

