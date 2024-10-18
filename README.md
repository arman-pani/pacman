# Deep Convolutional Q-Learning for Pac-Man

Welcome to the **Deep Convolutional Q-Learning for Pac-Man** project! This project implements a Deep Reinforcement Learning model using Convolutional Neural Networks (CNNs) to train an AI agent capable of playing the classic Atari game, Pac-Man, using **Deep Q-Learning** (DQN). This solution leverages OpenAI's Gym and Gymnasium environments.

[![Watch the video]()](https://github.com/arman-pani/pacman/blob/main/pacman_video.mp4)

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Training the Agent](#training-the-agent)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

Pac-Man is one of the most popular arcade games, and this project aims to train an AI agent to master the game using a combination of Convolutional Neural Networks and Q-Learning. Deep Q-Learning allows the agent to learn from game states and rewards, optimizing its decision-making over time.

The key objectives of this project are:
- To implement a reinforcement learning framework to play Pac-Man.
- To train a CNN-based Q-learning agent using the Pac-Man Atari environment.
- To achieve a balance between exploration and exploitation for effective learning.
  
## Getting Started

To get started, you will need to install the required libraries, including `gymnasium`, `atari-py`, and other dependencies that allow interaction with the Pac-Man environment. Follow the installation instructions below to set up your environment.

---

## Requirements

Ensure you have the following dependencies installed:
- Python 3.10+
- [Gymnasium](https://gymnasium.farama.org/)
- TensorFlow or PyTorch (for implementing the neural network)
- Numpy
- OpenAI Gym's Atari environments

**Additional Libraries**:
- `gymnasium[accept-rom-license, atari]` (for Atari game environments)
- `pygame` (for rendering the game)
- `box2d-py` (for simulation)

---

## Installation

To set up and run the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/pacman-dqn.git
   cd pacman-dqn

2. **Install Dependencies**:
   ```bash
   pip install gymnasium
   pip install "gymnasium[atari, accept-rom-license]"
   apt-get install -y swig
   pip install gymnasium[box2d]

3. **Download Atari ROMs**:
   ```bash
   python -m atari_py.import_roms /path/to/your/roms
   
4. **Verify Installation**:
   ```python
   import gymnasium as gym
   env = gym.make("ALE/Pacman-v5")
   env.reset()

---

## How It Works

### Deep Q-Learning
Deep Q-Learning (DQN) is a reinforcement learning algorithm that uses a neural network to approximate the Q-value function. The Q-value function predicts the expected future reward for taking a particular action in a given state. The neural network is trained on game frames (states) to minimize the difference between predicted Q-values and target Q-values, learned through gameplay.

### Convolutional Neural Network (CNN)
The CNN extracts features from raw pixel input (frames) of the Pac-Man game. These features represent game states, which the Q-network uses to decide on actions.

### Game Environment
We use the `Gymnasium` library, which provides access to the Pac-Man Atari environment:

- **State**: The screen pixel data of the game.
- **Action Space**: Possible movements of Pac-Man (up, down, left, right).
- **Rewards**: Positive rewards for collecting points, negative rewards for losing a life.

---

## Training the Agent

To train the DQN agent, follow these steps:

### 1. Initialize the Environment:
	```python
	import gymnasium as gym
	env = gym.make("ALE/Pacman-v5")
	state = env.reset()

### 2. Define the Neural Network:
Create a CNN-based Q-network to approximate the Q-value function. This network takes in the game state (a series of frames) and outputs Q-values for each possible action.

### 3. Training Loop:
- **Initialize replay memory** and store the agent’s experiences.
- For each episode:
  1. Get the current state.
  2. Choose an action based on the epsilon-greedy policy.
  3. Execute the action, receive a reward, and observe the next state.
  4. Store the transition in replay memory.
  5. Update the Q-network by sampling mini-batches from the replay memory.

### 4. Evaluate the Performance:
After training, evaluate the agent’s performance by letting it play several episodes and observe its score.

---

## Results

The training process produces a well-optimized agent that can play Pac-Man efficiently. The DQN agent learns to:

- Avoid ghosts.
- Collect pellets and power-ups.
- Maximize score through strategic movement.

Results of the trained agent, including performance graphs and gameplay videos, can be found in the `results` directory.

---

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes.
4. Submit a pull request.

Feel free to reach out if you have any questions or suggestions for improvements. Happy coding and happy gaming!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
