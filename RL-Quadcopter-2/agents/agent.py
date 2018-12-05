# From Udacity
import numpy as np
import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# From Udacity, changed architecture

from keras import layers, models, optimizers, regularizers
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        net = states
        # Add hidden layers
        net = layers.Dense(units=32, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001))(net)
        net = layers.LeakyReLU(alpha=0.3)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.4)(net)
        
        net = layers.Dense(units=1024, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(net)
        net = layers.LeakyReLU(alpha=0.3)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.4)(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        """注意，输出层生成的原始动作位于[0.0, 1.0]范围内（使用 sigmoid 激活函数）。
        因此，我们添加另一个层级，该层级会针对每个动作维度将每个输出缩放到期望的范围。
        这样会针对任何给定状态向量生成确定性动作。稍后将向此动作添加噪点，以生成某个探索性行为。"""
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions', kernel_initializer=layers.initializers.RandomUniform(minval=-0.001, maxval=0.001))(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        """另一个需要注意的是损失函数如何使用动作值（Q 值）梯度进行定义："""
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=1e-8)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        """这些梯度需要使用评论者模型计算，并在训练时提供梯度。因此指定为在训练函数中使用的“输入”的一部分："""
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

# From Udacity, changed architecture

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        """
        它在某些方面比行动者模型简单，但是需要注意几点。首先，行动者模型旨在将状态映射到动作，
        而评论者模型需要将（状态、动作）对映射到它们的 Q 值。这一点体现在了输入层中。"""
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        """It took me so, use_bias=True, bias_constraint=regularizers.l2(0.02)"""
        net_states = states
        net_states = layers.Dense(units=32, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001))(net_states)
        net_states = layers.LeakyReLU(alpha=0.3)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.4)(net_states)
        
        net_states = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001))(net_states)
        net_states = layers.LeakyReLU(alpha=0.3)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.4)(net_states)

        net_actions = actions
        net_actions = layers.Dense(units=32, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001))(net_actions)
        net_actions = layers.LeakyReLU(alpha=0.3)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.4)(net_actions)
        
        net_actions = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001))(net_actions)
        net_actions = layers.LeakyReLU(alpha=0.3)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.4)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        """这两个层级首先可以通过单独的“路径”（迷你子网络）处理，但是最终需要结合到一起
        例如，可以通过使用 Keras 中的 Add 层级类型来实现请参阅合并层级):"""
        net = layers.Add()([net_states, net_actions])
        
        net = layers.Dense(units=256, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001))(net)
        net = layers.LeakyReLU(alpha=0.3)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.4)(net)
        
        net = layers.Dense(units=16, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001))(net)
        net = layers.LeakyReLU(alpha=0.3)(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=1e-4)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        """该模型的最终输出是任何给定（状态、动作）对的 Q 值。但是，我们还需要计算此 Q 值相对于相应动作向量的梯度，以用于训练行动者模型。
        这一步需要明确执行，并且需要定义一个单独的函数来访问这些梯度："""
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

# noise solution inspired by lazem's github project: https://github.com/lazem/RL-Quadcopter-2/blob/2b228fcc0e271b41cb9021ef197266be9b950418/Quadcopter_Project.ipynb
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta=0.15, sigma=0.13):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        """Udacity's code is so bad!"""

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# From Udacity

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        """ADJUST HYPER-PARAMETERS"""
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        """注意，在用多个经验进行训练后，我们可以将新学习的权重（来自本地模型）复制到目标模型中。
        但是，单个批次可能会向该流程中引入很多偏差，因此最后进行软更新，由参数 tau 控制。"""
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        # Roll over last state and action
        self.last_state = next_state

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        """custom training function"""
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
    
    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[0:3]  # position only

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[0:3] = action  # linear force only
        return complete_action

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only


