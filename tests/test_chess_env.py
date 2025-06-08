import pytest
import chess
import numpy as np
from src.chess_env import ChessEnv


class TestChessEnv:
    def test_initial_state(self):
        env = ChessEnv()
        initial_planes = env.reset()
        assert initial_planes.shape == (21, 8, 8)
        assert env.board.fen() == chess.STARTING_FEN
        assert not env.is_game_over()

    def test_push_move(self):
        env = ChessEnv()
        move = chess.Move.from_uci("e2e4")
        state_planes, is_over, result = env.push_move(move)
        assert env.board.fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        assert state_planes.shape == (21, 8, 8)
        assert not is_over
        assert result == "*"

        # Test a game over scenario (Fool's Mate)
        env.reset()
        env.push_move(chess.Move.from_uci("f2f3"))
        env.push_move(chess.Move.from_uci("e7e5"))
        env.push_move(chess.Move.from_uci("g2g4"))
        state_planes, is_over, result = env.push_move(chess.Move.from_uci("d8h4"))
        assert is_over
        assert result == "0-1"  # Black wins

    def test_legal_moves(self):
        env = ChessEnv()
        initial_legal_moves = env.get_legal_moves()
        assert len(initial_legal_moves) == 20  # Standard initial legal moves

        env.push_move(chess.Move.from_uci("e2e4"))
        env.push_move(chess.Move.from_uci("e7e5"))
        legal_moves_after_e4e5 = env.get_legal_moves()
        assert len(legal_moves_after_e4e5) > 0  # Should still have legal moves

    def test_get_state_planes(self):
        env = ChessEnv()

        # Test piece positions (White Pawn at E2, then E4)
        initial_planes = env.get_state_planes()
        assert initial_planes[0, chess.square_rank(chess.E2), chess.square_file(chess.E2)] == 1  # White Pawn at E2

        env.push_move(chess.Move.from_uci("e2e4"))
        planes_after_e4 = env.get_state_planes()
        assert planes_after_e4[0, chess.square_rank(chess.E4), chess.square_file(chess.E4)] == 1  # White Pawn at E4
        assert planes_after_e4[0, chess.square_rank(chess.E2), chess.square_file(chess.E2)] == 0  # E2 is empty

        # Test player to move (White then Black)
        assert initial_planes[12, 0, 0] == 1  # White to move plane
        assert initial_planes[13, 0, 0] == 0  # Black to move plane
        assert planes_after_e4[12, 0, 0] == 0  # White to move plane
        assert planes_after_e4[13, 0, 0] == 1  # Black to move plane

        # Test castling rights (initial)
        assert initial_planes[14, 0, 0] == 1  # White King-side
        assert initial_planes[15, 0, 0] == 1  # White Queen-side
        assert initial_planes[16, 0, 0] == 1  # Black King-side
        assert initial_planes[17, 0, 0] == 1  # Black Queen-side

        # Test en passant (after d4, e.g.)
        env.reset()
        env.push_move(chess.Move.from_uci("e2e4"))
        env.push_move(chess.Move.from_uci("d7d5"))
        env.push_move(chess.Move.from_uci("e4d5"))
        env.push_move(chess.Move.from_uci("g8f6"))
        env.push_move(chess.Move.from_uci("d5d4"))
        env.push_move(chess.Move.from_uci("c7c5"))  # Creates en passant target on c6
        planes_ep = env.get_state_planes()
        assert env.board.ep_square == chess.C6
        assert planes_ep[18, chess.square_rank(chess.C6), chess.square_file(chess.C6)] == 1

        # Test halfmove clock and fullmove number (normalized)
        env.reset()
        # After 1. e4 e5 2. Nf3 Nc6
        env.push_move(chess.Move.from_uci("e2e4"))
        env.push_move(chess.Move.from_uci("e7e5"))
        env.push_move(chess.Move.from_uci("g1f3"))
        env.push_move(chess.Move.from_uci("b8c6"))

        planes_counters = env.get_state_planes()
        # Halfmove clock should be 2 after 1. e4 e5 2. Nf3 Nc6 (no captures/pawn moves)
        assert env.board.halfmove_clock == 2
        assert planes_counters[19, 0, 0] == 2 / 100.0  # Normalized

        # Fullmove number should be 3
        assert env.board.fullmove_number == 3
        assert planes_counters[20, 0, 0] == (3 - 1) / 2000.0  # Normalized (fullmove_number starts at 1)
