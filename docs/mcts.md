# AlphaChess Monte Carlo Tree Search (MCTS) Implementation

This document describes the Monte Carlo Tree Search (MCTS) algorithm as implemented in AlphaChess, which is crucial for guiding the agent's move selection during self-play and evaluation.

## 1. MCTS Overview

MCTS is a search algorithm that combines the generality of random simulation with the precision of tree search. In AlphaChess, it will leverage the neural network's policy and value outputs to efficiently explore the game tree.

## 2. MCTS Tree Structure

- **Nodes:** Each node in the MCTS tree represents a specific chess board state.
- **Edges:** Each edge connecting two nodes represents a legal move from the parent node's state to the child node's state.
- **Node Information:** Each node will store the following statistics:
  - `N(s, a)`: Visit count for taking action `a` from state `s`. This represents how many times this particular move has been explored.
  - `W(s, a)`: Total action value for taking action `a` from state `s`. This accumulates the values (outcomes) of all simulations that passed through this move.
  - `Q(s, a)`: Mean action value, calculated as `W(s, a) / N(s, a)`. This is the average outcome observed when taking action `a` from state `s`.
  - `P(s, a)`: Prior probability of taking action `a` from state `s`, as predicted by the neural network's policy head.

## 3. MCTS Phases

Each MCTS simulation consists of four phases:

### 3.1. Selection

- Starting from the root node, the algorithm iteratively selects child nodes until a leaf node (a node that has not yet been fully expanded) is reached.
- The selection policy balances exploration and exploitation using an Upper Confidence Bound (UCB) formula, typically UCB1 applied to the MCTS context:
  `UCB(s, a) = Q(s, a) + C * P(s, a) * sqrt(sum(N(s, b) for all b) / (1 + N(s, a)))`
  Where `C` is an exploration constant. This formula encourages selecting moves that have high predicted value (`Q`) and high prior probability (`P`), while also exploring less-visited moves.

### 3.2. Expansion

- When a leaf node is reached, if it is not a terminal state (e.g., checkmate or stalemate), it is expanded.
- All legal moves from this leaf node's state are generated.
- For each legal move, a new child node is created.
- The neural network is queried to predict the policy probabilities (`P`) for all legal moves from the current state and the value (`V`) of the current state. These `P` values are assigned as the initial prior probabilities for the newly created edges.

### 3.3. Simulation (Rollout)

- In AlphaZero-like MCTS, the "simulation" phase is replaced by the neural network's value prediction. The value `V` returned by the neural network's value head for the expanded node's state is used as the outcome of the simulation. This is a significant departure from traditional MCTS, which often involves random playouts to the end of the game.

### 3.4. Backpropagation

- The value `V` obtained from the simulation phase is propagated back up the tree along the path taken from the root to the expanded leaf node.
- For each node `(s, a)` on this path:
  - The visit count `N(s, a)` is incremented.
  - The total action value `W(s, a)` is updated by adding `V`. The sign of `V` is flipped for alternating players as the value is from the perspective of the player whose turn it is at state `s`.

## 4. Exploration during Self-Play

- To ensure sufficient exploration of the game tree during self-play, Dirichlet noise will be added to the prior probabilities `P(s, a)` at the root node of the MCTS tree. This encourages the agent to try a wider variety of moves, preventing it from getting stuck in local optima.

## 5. Configurable Parameters

- **Number of Simulations:** A critical configurable parameter will be the number of MCTS simulations performed for each move. A higher number of simulations generally leads to stronger move selection but requires more computational resources.

## 6. Resource Management

- **Memory Optimization:** MCTS trees can consume significant memory. Strategies such as pruning less promising branches or implementing a garbage collection mechanism for old nodes will be considered to manage memory usage, especially given the 16GB RAM constraint.
