import math
import chess
import torch
from typing import Dict, Optional, Tuple

from .chess_env import ChessEnv
from .nn_model import AlphaChessNet
from .move_encoder import MoveEncoderDecoder


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
        sum_N_s_b = self.N  # Corrected: N(s) is the visit count of the parent node itself

        # Debugging prints for UCB score calculation
        # print(
        #     f"  UCB Calc: move={child_move.uci()}, child_N={child_node.N}, "
        #     f"child_W={child_node.W}, child_Q={child_node.Q}"
        # )
        # print(f"  UCB Calc: root_turn={self.board.turn}, child_turn={child_node.board.turn}, q_value_flipped={q_value}")
        # print(f"  UCB Calc: prior_prob={prior_prob}, sum_N_s_b (parent N)={sum_N_s_b}, c_puct={c_puct}")

        exploration_term = (c_puct * prior_prob * math.sqrt(sum_N_s_b)) / (1 + child_node.N)
        ucb = q_value + exploration_term
        # print(f"  UCB Calc: final_ucb={ucb}")
        return ucb

    def select_child(self, c_puct: float) -> Tuple[Optional[chess.Move], Optional["MCTSNode"]]:
        best_move = None
        best_score = -float("inf")
        selected_child = None

        for move in self.legal_moves:
            if move not in self.children:
                prior_prob = self.P.get(move, 0.0) if self.P is not None else 0.0
                score = float("inf") if prior_prob > 0 else -float("inf")
            else:
                score = self.ucb_score(move, c_puct)

            if score > best_score:
                best_score = score
                best_move = move
                selected_child = self.children.get(best_move)

        if best_move is None and self.legal_moves:
            best_move = self.legal_moves[0]
            selected_child = self.children.get(best_move)

        return best_move, selected_child

    def expand(
        self,
        board: chess.Board,  # Added board parameter
        policy_logits: torch.Tensor,
        value: float,
        legal_moves: list,
        move_encoder: MoveEncoderDecoder,
    ):
        self.is_expanded = True

        policy_probs_tensor = torch.softmax(policy_logits, dim=-1).squeeze(0)

        # Map policy probabilities to legal moves using MoveEncoderDecoder
        self.P = {}
        for move in legal_moves:
            move_idx = move_encoder.encode(board, move)  # Pass board to encode
            if move_idx < len(policy_probs_tensor):  # Ensure index is within bounds
                self.P[move] = policy_probs_tensor[move_idx].item()
            else:
                self.P[move] = 0.0  # Should not happen if encoder is correct

        for move in legal_moves:
            if move not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(new_board, parent=self, move=move)

    def backpropagate(self, value: float, leaf_node_turn: chess.Color):
        node = self
        while node is not None:
            node.N += 1
            # Value is from the perspective of the player whose turn it was at the leaf node.
            # Flip value if the current node's turn is different from the leaf node's turn.
            if node.board.turn == leaf_node_turn:
                node.W += value
            else:
                node.W -= value
            node.Q = node.W / node.N
            node = node.parent


class MCTS:
    def __init__(
        self,
        model: AlphaChessNet,
        chess_env: ChessEnv,
        move_encoder: MoveEncoderDecoder,
        c_puct: float = 1.0,
        max_depth: int = 40,  # Added max_depth parameter, default 40 half-moves
    ):
        self.model = model
        self.chess_env = chess_env
        self.move_encoder = move_encoder
        self.c_puct = c_puct
        self.max_depth = max_depth

    def run_simulations(self, root_node: MCTSNode, num_simulations: int):
        for i in range(num_simulations):
            current_node = root_node
            path = [current_node]
            current_depth = 0

            # Selection
            while current_node.is_expanded and not current_node.board.is_game_over() and current_depth < self.max_depth:
                move, next_node = current_node.select_child(self.c_puct)
                if next_node is None:
                    break
                current_node = next_node
                path.append(current_node)
                current_depth += 1

            # If current_node became None due to break, skip this simulation
            if current_node is None:
                continue

            # Simulation (NN Evaluation) and Expansion
            if not current_node.board.is_game_over():
                # Temporarily set the env board to the current node's board for state plane generation
                original_board = self.chess_env.board.copy()
                self.chess_env.board = current_node.board.copy()
                board_state_planes = self.chess_env.get_state_planes()
                self.chess_env.board = original_board  # Restore original board

                # Convert to tensor and add batch dimension
                nn_input = torch.from_numpy(board_state_planes).float().unsqueeze(0)

                # Get policy and value from NN
                with torch.no_grad():
                    policy_logits, value = self.model(nn_input)

                # Expand the node
                current_node.expand(
                    current_node.board,  # Pass the board of the current node
                    policy_logits,
                    value.item(),
                    current_node.legal_moves,
                    self.move_encoder,
                )
                simulation_value = value.item()
            else:
                # Terminal node, use game result as value
                result = current_node.board.result()
                if result == "1-0":  # White wins
                    simulation_value = (
                        1.0 if current_node.board.turn == chess.BLACK else -1.0
                    )  # Value from perspective of player whose turn it is at current_node
                elif result == "0-1":  # Black wins
                    simulation_value = 1.0 if current_node.board.turn == chess.WHITE else -1.0
                else:  # Draw
                    simulation_value = 0.0

            # Backpropagation
            leaf_node_turn = current_node.board.turn
            current_node.backpropagate(simulation_value, leaf_node_turn)
