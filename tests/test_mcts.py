import pytest
import chess
import torch
import numpy as np
import math
from unittest.mock import Mock
from typing import Dict, List, Optional

from src.mcts import MCTSNode, MCTS
from src.chess_env import ChessEnv
from src.nn_model import AlphaChessNet
from src.move_encoder import MoveEncoderDecoder


class TestMCTSNode:
    """Tests for the MCTSNode class."""

    def test_node_init(self) -> None:
        """Tests the initialization of an MCTSNode."""
        board: chess.Board = chess.Board()
        node: MCTSNode = MCTSNode(board)
        assert node.board == board
        assert node.parent is None
        assert node.move is None
        assert node.N == 0
        assert node.W == 0
        assert node.Q == 0
        assert node.P is None
        assert not node.is_expanded
        assert len(node.legal_moves) == len(list(board.legal_moves))

        move: chess.Move = chess.Move.from_uci("e2e4")
        new_board: chess.Board = board.copy()
        new_board.push(move)
        child_node: MCTSNode = MCTSNode(new_board, parent=node, move=move)
        assert child_node.parent == node
        assert child_node.move == move

    def test_ucb_score(self) -> None:
        """Tests the UCB score calculation for child nodes."""
        board: chess.Board = chess.Board()
        root: MCTSNode = MCTSNode(board)
        root.N = 1  # Set root N to 1 for UCB calculation (N(s) in formula)

        # Mock policy probabilities for root
        root.P = {
            chess.Move.from_uci("e2e4"): 0.5,
            chess.Move.from_uci("d2d4"): 0.3,
            chess.Move.from_uci("g1f3"): 0.2,
        }

        # Create child nodes
        e2e4_move: chess.Move = chess.Move.from_uci("e2e4")
        e2e4_board: chess.Board = board.copy()
        e2e4_board.push(e2e4_move)
        e2e4_child: MCTSNode = MCTSNode(e2e4_board, parent=root, move=e2e4_move)
        root.children[e2e4_move] = e2e4_child

        d2d4_move: chess.Move = chess.Move.from_uci("d2d4")
        d2d4_board: chess.Board = board.copy()
        d2d4_board.push(d2d4_move)
        d2d4_child: MCTSNode = MCTSNode(d2d4_board, parent=root, move=d2d4_move)
        root.children[d2d4_move] = d2d4_child

        # Test unvisited child (should have infinite score if prior > 0)
        assert root.ucb_score(e2e4_move, c_puct=1.0) == float("inf")

        # Simulate some visits and values
        e2e4_child.N = 10
        e2e4_child.W = 5  # Value from child's perspective
        e2e4_child.Q = e2e4_child.W / e2e4_child.N  # 0.5

        d2d4_child.N = 5
        d2d4_child.W = -2  # Value from child's perspective
        d2d4_child.Q = d2d4_child.W / d2d4_child.N  # -0.4

        # The ucb_score method inverts Q-values if the child node's turn is different
        # from the current node's turn, ensuring values are from the current player's perspective.

        ucb_e2e4: float = root.ucb_score(e2e4_move, c_puct=1.0)
        ucb_d2d4: float = root.ucb_score(d2d4_move, c_puct=1.0)

        assert round(ucb_e2e4, 4) == -0.4545
        assert round(ucb_d2d4, 2) == 0.45

        # Test with root.N > 1
        root.N = 100
        ucb_e2e4_large_N: float = root.ucb_score(e2e4_move, c_puct=1.0)
        ucb_d2d4_large_N: float = root.ucb_score(d2d4_move, c_puct=1.0)
        assert round(ucb_e2e4_large_N, 4) == -0.0455
        assert round(ucb_d2d4_large_N, 2) == 0.9

        # Test with c_puct = 0 (pure exploitation)
        ucb_e2e4_c0: float = root.ucb_score(e2e4_move, c_puct=0.0)
        ucb_d2d4_c0: float = root.ucb_score(d2d4_move, c_puct=0.0)
        assert round(ucb_e2e4_c0, 1) == -0.5
        assert round(ucb_d2d4_c0, 1) == 0.4

        # Test with very large c_puct (pure exploration)
        large_c_puct: float = 1000.0
        ucb_e2e4_large_c: float = root.ucb_score(e2e4_move, c_puct=large_c_puct)
        ucb_d2d4_large_c: float = root.ucb_score(d2d4_move, c_puct=large_c_puct)
        assert round(ucb_e2e4_large_c, 3) == 454.045
        assert round(ucb_d2d4_large_c, 3) == 500.400

    def test_ucb_score_unvisited_positive_prior(self) -> None:
        """Tests UCB score for unvisited children with positive prior probabilities."""
        board: chess.Board = chess.Board()
        root: MCTSNode = MCTSNode(board)
        root.N = 1
        e2e4_move: chess.Move = chess.Move.from_uci("e2e4")
        root.P = {e2e4_move: 0.7}  # Positive prior

        # Child node does not exist in root.children, or N=0
        score: float = root.ucb_score(e2e4_move, c_puct=1.0)
        assert score == float("inf")

        # Child node exists but N=0
        e2e4_board: chess.Board = board.copy()
        e2e4_board.push(e2e4_move)
        e2e4_child: MCTSNode = MCTSNode(e2e4_board, parent=root, move=e2e4_move)
        root.children[e2e4_move] = e2e4_child
        e2e4_child.N = 0
        score_with_child_unvisited: float = root.ucb_score(e2e4_move, c_puct=1.0)
        assert score_with_child_unvisited == float("inf")

    def test_select_child(self) -> None:
        """Tests the selection of the child node with the highest UCB score."""
        board: chess.Board = chess.Board()
        root: MCTSNode = MCTSNode(board)
        root.N = 1  # Set root N to 1 for UCB calculation (N(s) in formula)

        root.P = {
            chess.Move.from_uci("e2e4"): 0.1,
            chess.Move.from_uci("d2d4"): 0.8,
            chess.Move.from_uci("g1f3"): 0.1,
        }

        # Create children
        e2e4_move: chess.Move = chess.Move.from_uci("e2e4")
        e2e4_board: chess.Board = board.copy()
        e2e4_board.push(e2e4_move)
        e2e4_child: MCTSNode = MCTSNode(e2e4_board, parent=root, move=e2e4_move)
        root.children[e2e4_move] = e2e4_child

        d2d4_move: chess.Move = chess.Move.from_uci("d2d4")
        d2d4_board: chess.Board = board.copy()
        d2d4_board.push(d2d4_move)
        d2d4_child: MCTSNode = MCTSNode(d2d4_board, parent=root, move=d2d4_move)
        root.children[d2d4_move] = d2d4_child

        g1f3_move: chess.Move = chess.Move.from_uci("g1f3")
        g1f3_board: chess.Board = board.copy()
        g1f3_board.push(g1f3_move)
        g1f3_child: MCTSNode = MCTSNode(g1f3_board, parent=root, move=g1f3_move)
        root.children[g1f3_move] = g1f3_child

        # Now expand the root node to populate children with initial Q and P values
        root.is_expanded = True
        root.P = {
            chess.Move.from_uci("e2e4"): 0.1,
            chess.Move.from_uci("d2d4"): 0.8,
            chess.Move.from_uci("g1f3"): 0.1,
        }
        # Manually add children to root.children for select_child to work
        root.children[e2e4_move] = e2e4_child
        root.children[d2d4_move] = d2d4_child
        root.children[g1f3_move] = g1f3_child

        # Simulate some visits and values (consistent with test_ucb_score)
        e2e4_child.N = 10
        e2e4_child.W = 5  # Value from child's perspective
        e2e4_child.Q = e2e4_child.W / e2e4_child.N  # 0.5

        d2d4_child.N = 5
        d2d4_child.W = -2  # Value from child's perspective
        d2d4_child.Q = d2d4_child.W / d2d4_child.N  # -0.4

        # Ensure g1f3_child is also visited so its UCB score is finite
        g1f3_child.N = 1
        g1f3_child.W = 0
        g1f3_child.Q = 0.0

        selected_move: Optional[chess.Move]
        selected_node: Optional[MCTSNode]
        selected_move, selected_node = root.select_child(c_puct=1.0)
        assert selected_move == d2d4_move
        assert selected_node == d2d4_child

        # Test tie-breaking for infinite scores (multiple unvisited children with positive priors)
        root_tie: MCTSNode = MCTSNode(board)
        root_tie.N = 1
        root_tie.P = {
            chess.Move.from_uci("a2a4"): 0.4,
            chess.Move.from_uci("b2b4"): 0.4,
            chess.Move.from_uci("c2c4"): 0.2,
        }
        # Create children for a2a4 and b2b4, but keep them unvisited
        a2a4_move: chess.Move = chess.Move.from_uci("a2a4")
        a2a4_board: chess.Board = board.copy()
        a2a4_board.push(a2a4_move)
        a2a4_child: MCTSNode = MCTSNode(a2a4_board, parent=root_tie, move=a2a4_move)
        root_tie.children[a2a4_move] = a2a4_child

        b2b4_move: chess.Move = chess.Move.from_uci("b2b4")
        b2b4_board: chess.Board = board.copy()
        b2b4_board.push(b2b4_move)
        b2b4_child: MCTSNode = MCTSNode(b2b4_board, parent=root_tie, move=b2b4_move)
        root_tie.children[b2b4_move] = b2b4_child

        c2c4_move: chess.Move = chess.Move.from_uci("c2c4")

        root_tie.P[a2a4_move] = 0.41
        root_tie.P[b2b4_move] = 0.40
        root_tie.P[c2c4_move] = 0.20

        selected_move_tie: Optional[chess.Move]
        selected_node_tie: Optional[MCTSNode]
        selected_move_tie, selected_node_tie = root_tie.select_child(c_puct=1.0)
        assert selected_move_tie == a2a4_move

        # Test fallback mechanism when no best move is found (e.g., no legal moves)
        empty_board: chess.Board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
        empty_node: MCTSNode = MCTSNode(empty_board)
        empty_node.is_expanded = True
        empty_node.P = {}
        empty_node.legal_moves = []

        selected_move_fallback: Optional[chess.Move]
        selected_node_fallback: Optional[MCTSNode]
        selected_move_fallback, selected_node_fallback = empty_node.select_child(c_puct=1.0)
        assert selected_move_fallback is None
        assert selected_node_fallback is None

    def test_select_child_fallback_no_prior(self) -> None:
        """Tests select_child behavior when no prior probabilities are available."""
        board: chess.Board = chess.Board()
        node: MCTSNode = MCTSNode(board)
        node.P = None  # P is None, so all prior_prob will be 0.0

        selected_move: Optional[chess.Move]
        selected_node: Optional[MCTSNode]
        selected_move, selected_node = node.select_child(c_puct=1.0)

        assert selected_move is None
        assert selected_node is None

    def test_expand(self) -> None:
        """Tests the expansion of an MCTS node using neural network outputs."""
        board: chess.Board = chess.Board()
        node: MCTSNode = MCTSNode(board)

        e2e4_move: chess.Move = chess.Move.from_uci("e2e4")
        d2d4_move: chess.Move = chess.Move.from_uci("d2d4")
        g1f3_move: chess.Move = chess.Move.from_uci("g1f3")
        a2a3_move: chess.Move = chess.Move.from_uci("a2a3")
        legal_moves: List[chess.Move] = [e2e4_move, d2d4_move, g1f3_move, a2a3_move]

        mock_move_encoder: Mock = Mock(spec=MoveEncoderDecoder)
        move_to_idx_map: Dict[chess.Move, int] = {
            chess.Move.from_uci("e2e4"): 0,
            d2d4_move: 1,
            g1f3_move: 2,
            a2a3_move: 10000,
        }
        mock_move_encoder.encode.side_effect = lambda b, move: move_to_idx_map.get(move, -1)

        mock_policy_logits: torch.Tensor = torch.tensor([[-1.0, 2.0, 0.5] + [-10.0] * (4672 - 3)])
        mock_value: torch.Tensor = torch.tensor([[0.5]])

        node.expand(board, mock_policy_logits, mock_value.item(), legal_moves, mock_move_encoder)

        assert node.is_expanded
        assert node.P is not None
        assert len(node.P) == len(legal_moves)

        full_policy_probs: List[float] = torch.softmax(mock_policy_logits.squeeze(0), dim=-1).tolist()

        assert math.isclose(node.P[e2e4_move], full_policy_probs[0], rel_tol=1e-6)
        assert math.isclose(node.P[d2d4_move], full_policy_probs[1], rel_tol=1e-6)
        assert math.isclose(node.P[g1f3_move], full_policy_probs[2], rel_tol=1e-6)

        assert node.P[a2a3_move] == 0.0

        assert len(node.children) == len(legal_moves)
        for move in legal_moves:
            assert move in node.children
            child: MCTSNode = node.children[move]
            assert child.parent == node
            assert child.move == move
            expected_child_board: chess.Board = board.copy()
            expected_child_board.push(move)
            assert child.board.fen() == expected_child_board.fen()

    def test_ucb_score_before_expansion(self) -> None:
        """Tests UCB score behavior for a node before it has been expanded."""
        board: chess.Board = chess.Board()
        node: MCTSNode = MCTSNode(board)
        assert node.P is None

        e2e4_move: chess.Move = chess.Move.from_uci("e2e4")
        assert e2e4_move in node.legal_moves

        score: float = node.ucb_score(e2e4_move, c_puct=1.0)
        assert score == -float("inf")

        e2e4_board: chess.Board = board.copy()
        e2e4_board.push(e2e4_move)
        e2e4_child: MCTSNode = MCTSNode(e2e4_board, parent=node, move=e2e4_move)
        node.children[e2e4_move] = e2e4_child
        e2e4_child.N = 0

        score_with_child_unvisited: float = node.ucb_score(e2e4_move, c_puct=1.0)
        assert score_with_child_unvisited == -float("inf")

    def test_select_child_before_expansion(self) -> None:
        """Tests select_child behavior for a node before it has been expanded."""
        board: chess.Board = chess.Board()
        node: MCTSNode = MCTSNode(board)
        assert node.P is None

        selected_move: Optional[chess.Move]
        selected_node: Optional[MCTSNode]
        selected_move, selected_node = node.select_child(c_puct=1.0)

        assert selected_move is None
        assert selected_node is None

    def test_backpropagate(self) -> None:
        """Tests the backpropagation of values and visit counts through the MCTS tree."""
        board: chess.Board = chess.Board()
        root: MCTSNode = MCTSNode(board)

        move1: chess.Move = chess.Move.from_uci("e2e4")
        board1: chess.Board = board.copy()
        board1.push(move1)
        child1: MCTSNode = MCTSNode(board1, parent=root, move=move1)
        root.children[move1] = child1

        move2: chess.Move = chess.Move.from_uci("e7e5")
        board2: chess.Board = board1.copy()
        board2.push(move2)
        child2: MCTSNode = MCTSNode(board2, parent=child1, move=move2)
        child1.children[move2] = child2

        simulation_value: float = -1.0
        leaf_node_turn: chess.Color = chess.BLACK

        child2.backpropagate(simulation_value, leaf_node_turn)

        assert child2.N == 1
        assert child2.W == 1.0
        assert child2.Q == 1.0

        assert child1.N == 1
        assert child1.W == -1.0
        assert child1.Q == -1.0

        assert root.N == 1
        assert root.W == 1.0
        assert root.Q == 1.0

        root.N, root.W, root.Q = 0, 0, 0
        child1.N, child1.W, child1.Q = 0, 0, 0
        child2.N, child2.W, child2.Q = 0, 0, 0

        simulation_value_draw: float = 0.0
        leaf_node_turn_draw: chess.Color = chess.WHITE

        child2.backpropagate(simulation_value_draw, leaf_node_turn_draw)
        assert child2.N == 1
        assert child2.W == 0.0
        assert child2.Q == 0.0
        assert child1.N == 1
        assert child1.W == 0.0
        assert child1.Q == 0.0
        assert root.N == 1
        assert root.W == 0.0
        assert root.Q == 0.0


class TestMCTS:
    """Tests for the MCTS class."""

    @pytest.fixture
    def mock_nn_model(self) -> Mock:
        """Provides a mock neural network model for MCTS tests."""
        mock_model: Mock = Mock(spec=AlphaChessNet)
        mock_model.return_value = (torch.randn(1, 4672), torch.tensor([[0.5]]))
        return mock_model

    @pytest.fixture
    def mock_chess_env(self) -> Mock:
        """Provides a mock ChessEnv for MCTS tests."""
        mock_env: Mock = Mock(spec=ChessEnv)
        mock_env.get_state_planes.return_value = np.zeros((21, 8, 8))
        mock_board: Mock = Mock(spec=chess.Board)
        mock_board.copy = Mock(return_value=chess.Board())
        mock_env.board = mock_board
        return mock_env

    @pytest.fixture
    def mock_move_encoder(self) -> Mock:
        """Provides a mock MoveEncoderDecoder for MCTS tests."""
        mock_encoder: Mock = Mock(spec=MoveEncoderDecoder)
        move_to_idx_map: Dict[chess.Move, int] = {
            chess.Move.from_uci("e2e4"): 0,
            chess.Move.from_uci("e7e5"): 1,
            chess.Move.from_uci("g1f3"): 2,
            chess.Move.from_uci("b8c6"): 3,
            chess.Move.from_uci("d2d4"): 4,
            chess.Move.from_uci("d7d5"): 5,
            chess.Move.from_uci("e4d5"): 6,
            chess.Move.from_uci("d8h4"): 7,
            chess.Move.from_uci("e1g1"): 8,
            chess.Move.from_uci("e8g8"): 9,
            chess.Move.from_uci("a2a4"): 10,
            chess.Move.from_uci("b7b5"): 11,
            chess.Move.from_uci("a4b5"): 12,
            chess.Move.from_uci("e5f6"): 13,
            chess.Move.from_uci("a7a8q"): 14,
            chess.Move.from_uci("a7a8r"): 15,
            chess.Move.from_uci("a7a8b"): 16,
            chess.Move.from_uci("a7a8n"): 17,
            chess.Move.from_uci("a2a1q"): 18,
            chess.Move.from_uci("a2a1r"): 19,
            chess.Move.from_uci("a2a1b"): 20,
            chess.Move.from_uci("a2a1n"): 21,
            chess.Move.from_uci("d4e3"): 22,
            chess.Move.from_uci("e5d6"): 23,
        }
        counter: int = len(move_to_idx_map)

        def encode_side_effect(board: chess.Board, move: chess.Move) -> int:
            nonlocal counter
            if move in move_to_idx_map:
                return move_to_idx_map[move]
            else:
                idx: int = counter
                move_to_idx_map[move] = idx
                counter += 1
                return idx

        mock_encoder.encode.side_effect = encode_side_effect
        mock_encoder.decode.side_effect = lambda idx: chess.Move(chess.A1, chess.A2)
        return mock_encoder

    def test_run_simulations_integration_and_deterministic_values(
        self, mock_nn_model: Mock, mock_chess_env: Mock, mock_move_encoder: Mock
    ) -> None:
        """Tests the MCTS run_simulations method with a single simulation and deterministic values."""
        board: chess.Board = chess.Board()
        root_node: MCTSNode = MCTSNode(board)

        e2e4_move: chess.Move = chess.Move.from_uci("e2e4")

        mock_move_encoder.encode.side_effect = lambda b, move: {
            e2e4_move: 0,
        }.get(move, 999)

        policy_logits_template: torch.Tensor = torch.full((1, 4672), -10.0)
        policy_logits_root: torch.Tensor = policy_logits_template.clone()
        policy_logits_root[0, mock_move_encoder.encode(board, e2e4_move)] = 5.0

        mock_nn_model.side_effect = [
            (policy_logits_root, torch.tensor([[0.8]])),  # Value for initial board
        ]

        mcts: MCTS = MCTS(mock_nn_model, mock_chess_env, mock_move_encoder, c_puct=1.0, max_depth=1)
        num_simulations: int = 1
        mcts.run_simulations(root_node, num_simulations)

        # Assertions for root node
        assert root_node.N == num_simulations
        assert root_node.is_expanded

        # Verify NN model calls
        assert mock_nn_model.call_count == num_simulations

        # Verify ChessEnv board management
        assert mock_chess_env.get_state_planes.call_count == num_simulations
        assert mock_move_encoder.encode.call_count >= num_simulations

        # Verify N, W, Q values for root after 1 simulation
        # Sim 1: Root (W) -> e2e4 (expand) -> NN(0.8) -> Backprop (e2e4: -0.8, Root: 0.8)
        # Root: N=1, W=0.8, Q=0.8

        assert math.isclose(root_node.N, 1)
        assert math.isclose(root_node.W, 0.8, rel_tol=1e-5)
        assert math.isclose(root_node.Q, 0.8, rel_tol=1e-5)

    def test_run_simulations_zero_simulations(
        self, mock_nn_model: Mock, mock_chess_env: Mock, mock_move_encoder: Mock
    ) -> None:
        """Tests MCTS run_simulations with zero simulations, ensuring no changes occur."""
        board: chess.Board = chess.Board()
        root_node: MCTSNode = MCTSNode(board)
        initial_N: int = root_node.N
        initial_W: float = root_node.W
        initial_Q: float = root_node.Q
        initial_is_expanded: bool = root_node.is_expanded

        mcts: MCTS = MCTS(mock_nn_model, mock_chess_env, mock_move_encoder)
        mcts.run_simulations(root_node, 0)

        assert root_node.N == initial_N
        assert root_node.W == initial_W
        assert root_node.Q == initial_Q
        assert root_node.is_expanded == initial_is_expanded
        mock_nn_model.assert_not_called()
        mock_chess_env.get_state_planes.assert_not_called()
        mock_move_encoder.encode.assert_not_called()

    def test_run_simulations_max_depth_selection_termination(
        self, mock_nn_model: Mock, mock_chess_env: Mock, mock_move_encoder: Mock
    ) -> None:
        """Tests MCTS run_simulations when max_depth is reached during selection."""
        board: chess.Board = chess.Board()
        root_node: MCTSNode = MCTSNode(board)

        # Manually expand root and its child to simulate a path where max_depth is reached during selection
        e2e4_move: chess.Move = chess.Move.from_uci("e2e4")
        e7e5_move: chess.Move = chess.Move.from_uci("e7e5")

        # Setup root node
        root_node.is_expanded = True
        root_node.P = {e2e4_move: 1.0}  # Strongly favor e2e4

        # Create e2e4 child and expand it
        board_e2e4: chess.Board = board.copy()
        board_e2e4.push(e2e4_move)
        e2e4_child: MCTSNode = MCTSNode(board_e2e4, parent=root_node, move=e2e4_move)
        root_node.children[e2e4_move] = e2e4_child
        e2e4_child.is_expanded = True
        e2e4_child.P = {e7e5_move: 1.0}  # Strongly favor e7e5

        # Create e7e5 child (not expanded yet, this will be the leaf)
        board_e2e4_e7e5: chess.Board = board_e2e4.copy()
        board_e2e4_e7e5.push(e7e5_move)
        e7e5_child: MCTSNode = MCTSNode(board_e2e4_e7e5, parent=e2e4_child, move=e7e5_move)
        e2e4_child.children[e7e5_move] = e7e5_child

        # Set max_depth to 2. Path: root (depth 0) -> e2e4 (depth 1) -> e7e5 (depth 2).
        # e7e5_child is at max_depth, so it should be expanded and backpropagated from.
        mcts: MCTS = MCTS(mock_nn_model, mock_chess_env, mock_move_encoder, c_puct=1.0, max_depth=2)
        num_simulations: int = 1

        # Mock NN output for the leaf node (e7e5_child)
        mock_nn_model.return_value = (torch.randn(1, 4672), torch.tensor([[0.7]]))  # Value for e7e5 board

        mcts.run_simulations(root_node, num_simulations)

        # Assertions
        assert root_node.N == 1
        assert e2e4_child.N == 1
        assert e7e5_child.N == 1

        assert e7e5_child.is_expanded  # The leaf node at max_depth should be expanded

        # Verify backpropagation
        # Value from NN is 0.7 for e7e5_child (White's turn).
        # e7e5_child (White's turn): N=1, W=0.7, Q=0.7
        # e2e4_child (Black's turn): N=1, W=-0.7, Q=-0.7
        # Root (White's turn): N=1, W=0.7, Q=0.7
        assert math.isclose(e7e5_child.W, 0.7, rel_tol=1e-6)
        assert math.isclose(e7e5_child.Q, 0.7, rel_tol=1e-6)
        assert math.isclose(e2e4_child.W, -0.7, rel_tol=1e-6)
        assert math.isclose(e2e4_child.Q, -0.7, rel_tol=1e-6)
        assert math.isclose(root_node.W, 0.7, rel_tol=1e-6)
        assert math.isclose(root_node.Q, 0.7, rel_tol=1e-6)

        # Verify NN was called once for the expansion of e7e5_child
        mock_nn_model.assert_called_once()

    def test_run_simulations_terminal_state_handling(
        self, mock_nn_model: Mock, mock_chess_env: Mock, mock_move_encoder: Mock
    ) -> None:
        """Tests MCTS run_simulations behavior when encountering terminal game states."""
        # Test Checkmate (White wins)
        checkmate_board_white_wins: chess.Board = chess.Board(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        )
        assert checkmate_board_white_wins.is_game_over()
        assert checkmate_board_white_wins.result() == "1-0"

        root_node_white_wins: MCTSNode = MCTSNode(checkmate_board_white_wins)
        mcts_white_wins: MCTS = MCTS(mock_nn_model, mock_chess_env, mock_move_encoder, c_puct=1.0, max_depth=1)
        mcts_white_wins.run_simulations(root_node_white_wins, 1)

        assert root_node_white_wins.N == 1
        # White wins (1-0), it's Black's turn at the terminal node. Black loses, so value is -1.0.
        assert math.isclose(root_node_white_wins.W, -1.0, rel_tol=1e-6)
        assert math.isclose(root_node_white_wins.Q, -1.0, rel_tol=1e-6)
        mock_nn_model.assert_not_called()  # NN should not be called for terminal nodes

        # Test Checkmate (Black wins)
        checkmate_board_black_wins: chess.Board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/5PPq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 2"
        )
        assert checkmate_board_black_wins.is_game_over()
        assert checkmate_board_black_wins.result() == "0-1"

        root_node_black_wins: MCTSNode = MCTSNode(checkmate_board_black_wins)
        mcts_black_wins: MCTS = MCTS(mock_nn_model, mock_chess_env, mock_move_encoder, c_puct=1.0, max_depth=1)
        mcts_black_wins.run_simulations(root_node_black_wins, 1)

        assert root_node_black_wins.N == 1
        # Black wins (0-1), it's White's turn at the terminal node. White loses, so value is -1.0.
        assert math.isclose(root_node_black_wins.W, -1.0, rel_tol=1e-6)
        assert math.isclose(root_node_black_wins.Q, -1.0, rel_tol=1e-6)

        # Test Stalemate (Draw)
        stalemate_board: chess.Board = chess.Board("8/8/8/8/8/8/7k/7K w - - 0 1")
        assert stalemate_board.is_game_over()
        assert stalemate_board.result() == "1/2-1/2"

        root_node_stalemate: MCTSNode = MCTSNode(stalemate_board)
        mcts_stalemate: MCTS = MCTS(mock_nn_model, mock_chess_env, mock_move_encoder, c_puct=1.0, max_depth=1)
        mcts_stalemate.run_simulations(root_node_stalemate, 1)

        assert root_node_stalemate.N == 1
        assert math.isclose(root_node_stalemate.W, 0.0, rel_tol=1e-6)
        assert math.isclose(root_node_stalemate.Q, 0.0, rel_tol=1e-6)

    def test_run_simulations_no_legal_moves_or_negative_inf_ucb(
        self, mock_nn_model: Mock, mock_chess_env: Mock, mock_move_encoder: Mock
    ) -> None:
        """Tests MCTS run_simulations when a node has no legal moves or all UCB scores are negative infinity."""
        # Scenario 1: Node with no legal moves (e.g., checkmate, but not necessarily game over from root's perspective)
        board_no_legal_moves: chess.Board = chess.Board(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        )
        assert board_no_legal_moves.is_game_over()
        assert board_no_legal_moves.result() == "1-0"
        assert not list(board_no_legal_moves.legal_moves)

        root_node_no_legal_moves: MCTSNode = MCTSNode(board_no_legal_moves)
        mcts_no_legal_moves: MCTS = MCTS(mock_nn_model, mock_chess_env, mock_move_encoder, c_puct=1.0, max_depth=1)
        mcts_no_legal_moves.run_simulations(root_node_no_legal_moves, 1)

        assert root_node_no_legal_moves.N == 1
        # White wins (1-0), it's Black's turn at the terminal node. Black loses, so value is -1.0.
        assert math.isclose(root_node_no_legal_moves.W, -1.0, rel_tol=1e-6)
        assert math.isclose(root_node_no_legal_moves.Q, -1.0, rel_tol=1e-6)

        # Scenario 2: Node where all legal moves lead to negative infinite UCB scores
        board_neg_inf_ucb: chess.Board = chess.Board()
        root_node_neg_inf_ucb: MCTSNode = MCTSNode(board_neg_inf_ucb)
        # No initial N, W, Q set for root or children, simulating a fresh start.
        # P will be None initially, leading to -inf UCB for all moves.

        e2e4_move: chess.Move = chess.Move.from_uci("e2e4")
        d2d4_move: chess.Move = chess.Move.from_uci("d2d4")
        legal_moves_neg_inf: List[chess.Move] = [e2e4_move, d2d4_move]
        root_node_neg_inf_ucb.legal_moves = legal_moves_neg_inf

        # Mock NN output for the first expansion (e2e4_child)
        policy_logits_template: torch.Tensor = torch.full((1, 4672), -10.0)
        policy_logits_e2e4: torch.Tensor = policy_logits_template.clone()
        policy_logits_e2e4[0, mock_move_encoder.encode(board_neg_inf_ucb, e2e4_move)] = 5.0
        mock_nn_model.side_effect = [
            (policy_logits_e2e4, torch.tensor([[0.5]])),  # Value for e2e4 board
        ]

        mcts_neg_inf_ucb: MCTS = MCTS(mock_nn_model, mock_chess_env, mock_move_encoder, c_puct=1.0, max_depth=1)
        mcts_neg_inf_ucb.run_simulations(root_node_neg_inf_ucb, 1)

        # After 1 simulation:
        # Root (White) selects e2e4 (first legal move, as all UCBs are -inf initially).
        # e2e4_child (Black) is expanded, gets value 0.5.
        # Backprop: Root gets 0.5, e2e4_child gets -0.5.
        e2e4_child_neg_inf: Optional[MCTSNode] = root_node_neg_inf_ucb.children.get(e2e4_move)

        assert root_node_neg_inf_ucb.N == 1
        assert math.isclose(root_node_neg_inf_ucb.W, 0.5, rel_tol=1e-6)
        assert math.isclose(root_node_neg_inf_ucb.Q, 0.5, rel_tol=1e-6)

        assert e2e4_child_neg_inf is not None
        assert (
            e2e4_child_neg_inf.N == 0
        )  # This child was created but not visited/backpropagated through in this simulation
        assert math.isclose(e2e4_child_neg_inf.W, 0.0, rel_tol=1e-6)  # Should be 0.0 as it wasn't backpropagated
        assert math.isclose(e2e4_child_neg_inf.Q, 0.0, rel_tol=1e-6)  # Should be 0.0 as it wasn't backpropagated
