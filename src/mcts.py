import math
import chess
import torch
from typing import Dict, Optional, Tuple

from .chess_env import ChessEnv
from .move_encoder import MoveEncoderDecoder


class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) tree.
    Each node corresponds to a specific board state.
    """

    def __init__(self, board: chess.Board, parent: Optional["MCTSNode"] = None, move: Optional[chess.Move] = None):
        """
        Initializes an MCTS node.

        Args:
            board: The chess.Board object representing the state of this node.
            parent: The parent MCTSNode of this node. None for the root node.
            move: The chess.Move that led to this board state from the parent.
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.N: int = 0  # Visit count
        self.W: float = 0.0  # Total action-value
        self.Q: float = 0.0  # Mean action-value (W/N)
        self.P: Optional[Dict[chess.Move, float]] = None  # Policy probabilities from the neural network
        self.is_expanded = False  # True if this node has been expanded by the neural network
        self.legal_moves = list(board.legal_moves)

    def ucb_score(self, child_move: chess.Move, c_puct: float) -> float:
        """
        Calculates the UCB (Upper Confidence Bound) score for a child node.

        Args:
            child_move: The move leading to the child node.
            c_puct: A constant determining the level of exploration.

        Returns:
            The UCB score for the child node.
        """
        child_node = self.children.get(child_move)
        prior_prob = self.P.get(child_move, 0.0) if self.P is not None else 0.0

        if child_node is None or child_node.N == 0:
            # For unvisited nodes, return a high value based on prior probability
            # to encourage exploration. If prior_prob is 0, return -inf to avoid selection.
            return float("inf") if prior_prob > 0 else -float("inf")

        q_value = child_node.Q
        # Invert Q-value if the child node's turn is different from the current node's turn
        # because Q-values are from the perspective of the player to move at the child node.
        if self.board.turn != child_node.board.turn:
            q_value = -q_value

        parent_total_visits = self.N

        c_prior_prod = c_puct * prior_prob
        exploration_term = c_prior_prod * math.sqrt(parent_total_visits) / (1 + child_node.N)
        ucb = q_value + exploration_term
        return ucb

    def select_child(self, c_puct: float) -> Tuple[Optional[chess.Move], Optional["MCTSNode"]]:
        """
        Selects the child node with the highest UCB score.

        Args:
            c_puct: A constant determining the level of exploration.

        Returns:
            A tuple containing the selected move and the corresponding MCTSNode.
            Returns (None, None) if there are no legal moves.
        """
        if not self.legal_moves:
            return None, None

        best_move = None
        best_score = -float("inf")
        best_prior_prob = -1.0  # Used for tie-breaking when scores are infinite

        for move in self.legal_moves:
            current_score = self.ucb_score(move, c_puct)

            # Select the child with the highest UCB score.
            # If scores are equal and infinite, prefer the one with higher prior probability.
            # Note: The ucb_score method handles returning -inf for unvisited nodes with 0 prior.
            # This ensures that only nodes with positive prior or already visited nodes are considered
            # for selection when multiple nodes have infinite UCB scores.
            if current_score > best_score:
                best_score = current_score
                best_move = move
                # Update best_prior_prob only if current_score is infinite, for tie-breaking
                if current_score == float("inf"):
                    best_prior_prob = self.P.get(move, 0.0) if self.P is not None else 0.0
            elif current_score == best_score and current_score == float("inf"):
                prior_prob_for_move = self.P.get(move, 0.0) if self.P is not None else 0.0
                if prior_prob_for_move > best_prior_prob:
                    best_prior_prob = prior_prob_for_move
                    best_move = move
            # If scores are equal and finite, the first encountered is fine (arbitrary tie-break)

        selected_child = self.children.get(best_move) if best_move else None
        return best_move, selected_child

    def expand(
        self,
        board: chess.Board,
        policy_logits: torch.Tensor,
        value: float,
        legal_moves: list[chess.Move],
        move_encoder: MoveEncoderDecoder,
    ):
        self.is_expanded = True

        policy_probs_tensor = torch.softmax(policy_logits, dim=-1).squeeze(0)

        self.P = {}
        for move in legal_moves:
            move_idx = move_encoder.encode(board, move)
            if move_idx < len(policy_probs_tensor):
                self.P[move] = policy_probs_tensor[move_idx].item()
            else:
                self.P[move] = 0.0

        for move in legal_moves:
            if move not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(new_board, parent=self, move=move)

    def backpropagate(self, value: float, leaf_node_turn: chess.Color):
        node = self
        while node is not None:
            node.N += 1
            if node.board.turn == leaf_node_turn:
                node.W += value
            else:
                node.W -= value
            node.Q = node.W / node.N
            node = node.parent


class MCTS:
    """
    Implements the Monte Carlo Tree Search algorithm.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        chess_env: ChessEnv,
        move_encoder: MoveEncoderDecoder,
        c_puct: float = 1.0,
        max_depth: int = 40,
    ):
        """
        Initializes the MCTS.

        Args:
            model: The neural network model (e.g., AlphaChessNet) for policy and value prediction.
            chess_env: The chess environment.
            move_encoder: The move encoder/decoder.
            c_puct: Exploration constant for UCB.
            max_depth: Maximum depth for MCTS simulations to prevent infinite loops.
        """
        self.model = model
        self.chess_env = chess_env
        self.move_encoder = move_encoder
        self.c_puct = c_puct
        self.max_depth = max_depth

    def add_dirichlet_noise_to_root(self, root_node: MCTSNode, alpha: float, epsilon: float):
        """
        Adds Dirichlet noise to the policy probabilities of the root node.
        This encourages exploration in the early stages of the search.

        Args:
            root_node: The root node of the MCTS tree.
            alpha: Parameter for the Dirichlet distribution.
            epsilon: Weight of the Dirichlet noise.
        """
        if root_node.P is None:
            return

        moves = list(root_node.P.keys())
        if not moves:
            return

        if epsilon == 0.0:  # If epsilon is 0, no noise is added, so return early
            return

        # Ensure alpha is strictly positive for Dirichlet distribution
        safe_alpha = max(alpha, 1e-5)
        dirichlet_dist = torch.distributions.dirichlet.Dirichlet(torch.full((len(moves),), safe_alpha))
        noise = dirichlet_dist.sample()

        for i, move in enumerate(moves):
            root_node.P[move] = (1 - epsilon) * root_node.P[move] + epsilon * noise[i].item()

        # Re-normalize probabilities after adding noise
        sum_p = sum(root_node.P.values())
        if sum_p > 0:
            for move in moves:
                root_node.P[move] /= sum_p

    def run_simulations(self, root_node: MCTSNode, num_simulations: int):
        """
        Runs a specified number of MCTS simulations starting from the root node.

        Each simulation involves:
        1. Selection: Traverse the tree by selecting the child with the highest UCB score.
        2. Expansion: When a leaf node is reached, if it's not a terminal state,
                      use the neural network to predict policy and value, and expand the node.
        3. Simulation (Rollout): The value from the neural network is used directly.
        4. Backpropagation: Update visit counts and Q-values along the traversed path.

        Args:
            root_node: The starting node for simulations.
            num_simulations: The number of simulations to perform.
        """
        for i in range(num_simulations):
            current_node = root_node
            path = [current_node]
            current_depth = 0

            # Selection phase
            game_is_over = current_node.board.is_game_over()
            while current_node.is_expanded and not game_is_over and current_depth < self.max_depth:
                move, next_node = current_node.select_child(self.c_puct)
                if next_node is None:  # No legal moves from this node, or selection failed
                    break
                current_node = next_node
                path.append(current_node)
                current_depth += 1

            # If the loop broke because current_node became None (shouldn't happen with
            # select_child returning None,None) or if it's a terminal node, handle it.
            # The 'if current_node is None: continue' is removed as select_child ensures
            # a node is returned or loop breaks.

            # Expansion and Evaluation phase
            if not current_node.board.is_game_over():
                # Temporarily set the environment's board to the current node's board
                # to get state planes for NN input. This is necessary because ChessEnv
                # operates on its internal board state.
                original_board = self.chess_env.board.copy()
                self.chess_env.board = current_node.board.copy()
                board_state_planes = self.chess_env.get_state_planes()
                self.chess_env.board = original_board  # Restore original board

                nn_input = torch.from_numpy(board_state_planes).float().unsqueeze(0)

                with torch.no_grad():
                    policy_logits, value = self.model(nn_input)

                current_node.expand(
                    current_node.board,
                    policy_logits,
                    value.item(),
                    current_node.legal_moves,
                    self.move_encoder,
                )
                simulation_value = value.item()
            else:
                # Terminal node: determine value based on game result
                result = current_node.board.result()
                if result == "1-0":  # White wins
                    simulation_value = 1.0 if current_node.board.turn == chess.WHITE else -1.0
                elif result == "0-1":  # Black wins
                    simulation_value = 1.0 if current_node.board.turn == chess.BLACK else -1.0
                else:  # Draw
                    simulation_value = 0.0

            # Backpropagation phase
            leaf_node_turn = current_node.board.turn
            current_node.backpropagate(simulation_value, leaf_node_turn)
