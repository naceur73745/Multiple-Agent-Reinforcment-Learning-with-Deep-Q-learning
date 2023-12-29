import numpy as np 

class ReplayBuffer:
    def __init__(self, capacity, input_dim, n_actions):
        """
        A replay buffer class used in reinforcement learning for storing and sampling experiences.

        Args:
        - capacity (int): Maximum capacity of the replay buffer.
        - input_dim (int): Dimensionality of the input state.
        - n_actions (int): Number of possible actions in the environment.
        """
        self.capacity = capacity
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.mem_cntr = 0

        # Arrays to store transitions
        self.states = np.zeros((self.capacity, self.input_dim))
        self.next_states = np.zeros((self.capacity, self.input_dim))
        self.actions = np.zeros((self.capacity, self.n_actions))
        self.rewards = np.zeros(self.capacity)
        self.dones = np.zeros(self.capacity, dtype=bool)

    def store_transition(self, state, next_state, action, reward, done):
        """
        Store a transition (state, next_state, action, reward, done) in the replay buffer.

        Args:
        - state (array-like): Current state.
        - next_state (array-like): Next state after taking an action.
        - action (array-like): Action taken in the current state.
        - reward (float): Reward received after taking an action.
        - done (bool): Whether the episode is done or not.
        """
        index = self.mem_cntr % self.capacity
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.mem_cntr += 1

    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
        - batch_size (int): Number of transitions to sample.

        Returns:
        - Tuple containing arrays of states, next_states, actions, rewards, and dones.
        """
        max_mem = min(self.mem_cntr, self.capacity)

        if max_mem >= batch_size:
            batch_indices = np.random.choice(max_mem, batch_size, replace=False)
        else:
            batch_indices = np.random.choice(max_mem, batch_size, replace=True)

        states = self.states[batch_indices]
        next_states = self.next_states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        dones = self.dones[batch_indices]

        return states, next_states, actions, rewards, dones
