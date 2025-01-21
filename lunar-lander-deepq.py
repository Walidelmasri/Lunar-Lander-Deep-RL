# If you encounter errors due to version conflicts, you can install gym with the following command:
# pip install gym[box2d]==0.25.5

# Import necessary libraries for deep learning, reinforcement learning, and visualization
import random
from base64 import b64encode
from collections import deque, namedtuple
from time import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from IPython.display import HTML
from pyvirtualdisplay import Display

# Set the style for Matplotlib plots
plt.style.use('fivethirtyeight')

# Initialize a virtual display for rendering the environment without a physical monitor
display = Display(visible=False, size=(1400, 900))
_ = display.start()
video_name = "landerVideo.mp4"

# Function to render and display MP4 videos inline in a notebook
def render_mp4(videopath):
    """
    Encodes an MP4 video as a base64 string for inline display in a notebook.
    """
    mp4 = open(videopath, 'rb').read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    return f'<video width=400 controls><source src="data:video/mp4;' \
           f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'

# Function to choose the computation device: GPU if available, otherwise CPU
def chooseDevice(model=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Use data parallelization if multiple GPUs are available
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Multiple GPUs available ({torch.cuda.device_count()})... Using data parallelization for {model}")
        model = nn.DataParallel(model)
        
    return device

# Choose device and initialize the LunarLander-v2 environment
device = chooseDevice()
env = gym.make("LunarLander-v2")

# Extract action and observation space sizes
state_size = env.action_space.n  # Number of discrete actions
action_size = env.observation_space.shape[0]  # Size of the observation space
print(f"Action Space size: {state_size}")
print(f"Observation Space Size: {action_size}")

# Set up a video recorder for capturing gameplay
video = VideoRecorder(env, video_name)

# Define a neural network for the Q-Learning agent
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        A fully connected feedforward neural network for Q-learning.
        
        Parameters:
        - state_size (int): Size of the input (state dimensions)
        - action_size (int): Size of the output (action dimensions)
        - seed (int): Random seed for reproducibility
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)  # First hidden layer
        self.fc2 = nn.Linear(32, 256)        # Second hidden layer
        self.fc3 = nn.Linear(256, action_size)  # Output layer
    
    def forward(self, x):
        """Defines the forward pass through the network."""
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        x = self.fc3(x)          # Output layer (no activation here)
        return x

# Replay buffer for storing past experiences
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Initializes a replay buffer for sampling experiences during training.
        """
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)  # Stores a fixed number of experiences
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Adds an experience to the replay buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
                
    def sample(self):
        """
        Samples a batch of experiences from the buffer.
        Returns states, actions, rewards, next states, and done flags as tensors.
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        """Returns the current size of the replay buffer."""
        return len(self.memory)

# Define training parameters for the DQN agent
BUFFER_SIZE = int(1e5)  # Maximum size of the replay buffer
BATCH_SIZE = 64         # Number of samples per training batch
GAMMA = 0.99            # Discount factor for future rewards
TAU = 1e-3              # Soft update parameter for the target network
LR = 1e-4               # Learning rate for the optimizer
UPDATE_EVERY = 4        # Frequency of network updates

# Epsilon-greedy policy parameters
EPS_START = 1.0      # Initial epsilon value
EPS_DECAY = 0.995    # Epsilon decay rate
EPS_MIN = 0.01       # Minimum epsilon

# Plotting the effect of different epsilon decay rates
plt.figure(figsize=(10, 6))
for decay_rate in [0.9, 0.99, 0.991, 0.9999]:
    test_eps = EPS_START
    eps_list = []
    for _ in range(1000):
        test_eps = max(test_eps * decay_rate, EPS_MIN)
        eps_list.append(test_eps)
    plt.plot(eps_list, label=f'Decay rate: {decay_rate}')

plt.title('Epsilon Decay Schedules')
plt.legend(loc='best')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.show()
