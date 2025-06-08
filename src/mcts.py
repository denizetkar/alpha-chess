import math
import chess
import torch
from typing import Dict, Optional, Tuple

from .chess_env import ChessEnv
from .move_encoder import MoveEncoderDecoder

# AlphaChessNet is only used for type hinting in MCTS, but MCTS now accepts torch.nn.Module
# from .nn_model import AlphaChessNet


class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move: Optional[chess.Move] = None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: Optional[Dict[chess.Move, float]] = None
        self.is_expanded = False
        self.legal_moves = list(board.legal_moves)

    def ucb_score(self, child_move: chess.Move, c_puct: float) -> float:
        child_node = self.children.get(child_move)

        if child_node is None or child_node.N == 0:
            prior_prob = self.P.get(child_move, 0.0) if self.P is not None else 0.0
            return float("inf") if prior_prob > 0 else -float("inf")

        q_value = child_node.Q
        if self.board.turn != child_node.board.turn:
            q_value = -q_value

        prior_prob = self.P.get(child_move, 0.0) if self.P is not None else 0.0
        sum_N_s_b = self.N

        c_prior_prod = c_puct * prior_prob
        exploration_term = c_prior_prod * math.sqrt(sum_N_s_b) / (1 + child_node.N)
        ucb = q_value + exploration_term
        return ucb

    def select_child(self, c_puct: float) -> Tuple[Optional[chess.Move], Optional["MCTSNode"]]:
        best_move = None
        best_score = -float("inf")
        best_prior_prob = -1.0

        for move in self.legal_moves:
            prior_prob_for_move = self.P.get(move, 0.0) if self.P is not None else 0.0

            if move not in self.children:
                current_score = float("inf") if prior_prob_for_move > 0 else -float("inf")
            else:
                current_score = self.ucb_score(move, c_puct)

            # Logic for selecting the best move, including tie-breaking for infinite scores
            if current_score > best_score:
                best_score = current_score
                best_move = move
                best_prior_prob = prior_prob_for_move
            elif current_score == best_score and current_score == float("inf"):
                if prior_prob_for_move > best_prior_prob:
                    best_prior_prob = prior_prob_for_move
                    best_move = move

        selected_child = self.children.get(best_move) if best_move else None

        if best_move is None and self.legal_moves:
            best_move = self.legal_moves[0]
            selected_child = self.children.get(best_move)

        return best_move, selected_child

    def expand(
        self,
        board: chess.Board,
        policy_logits: torch.Tensor,
        value: float,
        legal_moves: list,
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
    def __init__(
        self,
        model: torch.nn.Module,
        chess_env: ChessEnv,
        move_encoder: MoveEncoderDecoder,
        c_puct: float = 1.0,
        max_depth: int = 40,
    ):
        self.model = model
        self.chess_env = chess_env
        self.move_encoder = move_encoder
        self.c_puct = c_puct
        self.max_depth = max_depth

    def add_dirichlet_noise_to_root(self, root_node: MCTSNode, alpha: float, epsilon: float):
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

        sum_p = sum(root_node.P.values())
        if sum_p > 0:
            for move in moves:
                root_node.P[move] /= sum_p

    def run_simulations(self, root_node: MCTSNode, num_simulations: int):
        for i in range(num_simulations):
            current_node = root_node
            path = [current_node]
            current_depth = 0

            while current_node.is_expanded and not current_node.board.is_game_over() and current_depth < self.max_depth:
                move, next_node = current_node.select_child(self.c_puct)
                if next_node is None:
                    break
                current_node = next_node
                path.append(current_node)
                current_depth += 1

            if current_node is None:
                continue

            if not current_node.board.is_game_over():
                original_board = self.chess_env.board.copy()
                self.chess_env.board = current_node.board.copy()
                board_state_planes = self.chess_env.get_state_planes()
                self.chess_env.board = original_board

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
                result = current_node.board.result()
                if result == "1-0":  # White wins
                    simulation_value = 1.0 if current_node.board.turn == chess.WHITE else -1.0
                elif result == "0-1":  # Black wins
                    simulation_value = 1.0 if current_node.board.turn == chess.BLACK else -1.0
                else:  # Draw
                    simulation_value = 0.0

            leaf_node_turn = current_node.board.turn
            current_node.backpropagate(simulation_value, leaf_node_turn)
