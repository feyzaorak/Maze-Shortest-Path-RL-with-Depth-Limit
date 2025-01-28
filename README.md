# Maze Shortest Path Planning with Depth Limited Reinforcement Learning 

## Project Overview
This project implements a reinforcement learning environment for solving a maze navigation problem. To achieve more efficiency without sacrificing accuracy, current state dictionary and debt limit convergence are used. The goal is to generate random mazes, train an agent using reinforcement learning, and enable the agent to find the optimal path from a any point to an endpoint.  

## Features
- **Random Maze Generation**: Dynamically generates mazes with varying structures and obstacle placements.
- **Random Start States**: Initializes random starting positions for the agent at the beginning of every epoch, ensuring diverse training scenarios.
- **State Graph with Valid Moves**: Represents the maze as a graph where each cell is a node, and valid movements are edges. Uses a dictionary to create a valid state graph, avoiding walls and blocks, ensuring movements are restricted to traversable paths.
- **Reinforcement Learning Algorithms**: Integrates Deep Q-Network (DQN) for training the agent.
- **Depth Limit**: In each epoch, efficiency is achieved by setting limits on the steps.
- **GPU Support**: Leverages PyTorch to utilize CUDA if available for accelerated computation.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Numpy
- Matplotlib
- Jupyter Notebook

### Installation Steps
1. Clone this repository or download the source code.
2. Install dependencies:
   ```bash
   pip install torch numpy matplotlib
   ```
3. Open the notebook in Jupyter:
   ```bash
   jupyter notebook reinforcement_maze.ipynb
   ```

---

## File Structure
- **depth_limit_maze.ipynb** - Main notebook implementing maze generation and reinforcement learning.
- **images/** - Optional folder for storing images.

---

## Code Walkthrough

### 1. Environment Setup
- **Device Configuration:**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
  Ensures compatibility with GPUs if available.

- **Constants:**
  ```python
  LEFT = 0
  UP = 1
  RIGHT = 2
  DOWN = 3
  actions_dict = {LEFT: 'left', UP: 'up', RIGHT: 'right', DOWN: 'down'}
  ```
  Defines actions for moving through the maze.
  
- **Creating Random Maze:**
  ```python
  def create_maze(rows, cols, wall_prob=0.3):
      maze = np.random.choice([0, 1], size=(rows, cols), p=[wall_prob, 1 - wall_prob])
      maze[0][0] = 1  # Start point
      maze[rows - 1][cols - 1] = 1  # End point
      return maze
  ```
  Generates a random maze based on specified dimensions and wall probabilities.

### 2. Maze Representation with Valid-State Dictionary
- **Graph Structure:**
  ```python
  class MazeGraph:
      def __init__(self, maze):
          self._maze = np.array(maze)
          self.rows, self.cols = self._maze.shape
          self.graph = defaultdict(list)
          self.actions = [LEFT, UP, RIGHT, DOWN]  
          self.build_graph()
  ```
  Creates a graph representation of the maze where nodes represent cells, and edges represent valid moves.

- **Validation:**
  ```python
  def get_valid_actions(self, current_state):
        valid_moves = self.graph[current_state]
        return [action for _, action in valid_moves]
  ```
  Ensures movements stay within bounds and avoid obstacles by using a dictionary-based state graph to store valid moves.

### 3. Reinforcement Learning Algorithm
- **Agent Training and Action Selection:**
  Includes placeholder code for implementing DQN.
  ```python
  class QLearningAgent:
      def __init__(self, ...):
          # Initialize agent parameters
          pass
  ```
- **Depth Limit:**
  It divides the problem into subproblems by limiting the agent's steps in each iteration to twice the size of the maze and tries to approach the global optimum with local optima. It was expected to reach accuracy while increasing efficiency in larger and more complex environments. The results of the tests are as follows:

- **Training Loop:**
  Iterates through episodes to update policies and learn optimal paths. Random start states are used in each epoch to enhance exploration and generalization.

- **Visualization:**
  ```python
  plt.imshow(maze, cmap='gray')
  ```
  Visualizes the maze and the agent's path.

---

## Usage
1. **Define Maze Input:** Provide a binary matrix where `1` represents a free path and `0` represents walls.
2. **Run Training:** Execute cells step-by-step to initialize the environment and train the agent.
3. **Test the Model:** Evaluate the trained agent's performance on new mazes.

---

## Configuration
- Adjust maze size, randomness, and wall density in the initial setup cells.
- Modify training parameters like learning rate, discount factor, and exploration rate in the agent class.

---

## References

[Q-Learning Maze Example](https://www.samyzaf.com/ML/rl/qmaze.html#Deep-Reinforcement-Learning-for-Maze-Solving
)

[Deep Q-Learning Maze Repository](https://github.com/giorgionicoletti/deep_Q_learning_maze/tree/master)

[Maze Pathfinding Example](https://github.com/HoomanRamezani/rl-maze-pathfinding/blob/main/data-generation/QMaze_Data.ipynb)

