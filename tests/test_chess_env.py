import chess
from src.chess_env import (
    ChessEnv,
    NUM_PLANES,
    WHITE_PIECE_PLANES_START,
    BLACK_PIECE_PLANES_START,
    WHITE_TO_MOVE_PLANE,
    BLACK_TO_MOVE_PLANE,
    WHITE_KINGSIDE_CASTLING_PLANE,
    WHITE_QUEENSIDE_CASTLING_PLANE,
    BLACK_KINGSIDE_CASTLING_PLANE,
    BLACK_QUEENSIDE_CASTLING_PLANE,
    EN_PASSANT_PLANE,
    FIFTY_MOVE_PLANE,
    FULLMOVE_PLANE,
    HISTORY_PLANES_START,
    NUM_HISTORY_PLANES,
    FIFTY_MOVE_NORMALIZATION_FACTOR,
    FULLMOVE_NORMALIZATION_FACTOR,
)


class TestChessEnv:
    """Tests for the ChessEnv class."""

    def test_initial_state(self):
        """Tests the initial state of the chess environment."""
        env = ChessEnv()
        initial_planes = env.reset()
        assert initial_planes.shape == (NUM_PLANES, 8, 8)
        assert env.board.fen() == chess.STARTING_FEN
        assert not env.is_game_over()

    def test_reset_history(self):
        """Tests that the board history is correctly reset."""
        env = ChessEnv()
        # Push some moves to build up history
        env.push_move(chess.Move.from_uci("e2e4"))
        env.push_move(chess.Move.from_uci("e7e5"))
        env.push_move(chess.Move.from_uci("g1f3"))
        assert len(env.board_history) == 3

        # Reset the environment
        env.reset()

        # Verify history is cleared and then contains only the initial board
        assert len(env.board_history) == 1
        assert env.board_history[0].fen() == chess.STARTING_FEN
        assert env.board.fen() == chess.STARTING_FEN

    def test_board_history_management(self):
        """Tests the management of the board history, including maxlen and immutability."""
        env = ChessEnv()
        # Push more than NUM_HISTORY_PLANES moves to test maxlen
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d4", "e5d4", "c3d4"]
        for move_uci in moves:
            env.push_move(chess.Move.from_uci(move_uci))

        # Verify maxlen=NUM_HISTORY_PLANES
        assert len(env.board_history) == NUM_HISTORY_PLANES

        # Verify that board.copy() is used (immutability)
        # Get a board from history
        historical_board = env.board_history[0]
        # Make a change to the current board
        env.board.push(chess.Move.from_uci("d4e5"))
        # Ensure the historical board remains unchanged
        assert historical_board.fen() != env.board.fen()

        # Let's reset and build history carefully to test planes.
        env.reset()
        test_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"]
        for move_uci in test_moves:
            env.push_move(chess.Move.from_uci(move_uci))

        planes = env.get_state_planes()

        expected_turns = [
            chess.BLACK,  # after e2e4
            chess.WHITE,  # after e7e5
            chess.BLACK,  # after g1f3
            chess.WHITE,  # after b8c6
            chess.BLACK,  # after f1c4
            chess.WHITE,  # after f8c5
            chess.BLACK,  # after c2c3
            chess.WHITE,  # after g8f6
        ]

        for i, expected_turn in enumerate(expected_turns):
            # Check if the plane is all ones (White to move) or all zeros (Black to move)
            if expected_turn == chess.WHITE:
                assert (
                    planes[HISTORY_PLANES_START + i, :, :] == 1
                ).all(), f"Plane {HISTORY_PLANES_START + i} should be all ones (White)"
            else:
                assert (
                    planes[HISTORY_PLANES_START + i, :, :] == 0
                ).all(), f"Plane {HISTORY_PLANES_START + i} should be all zeros (Black)"

    def test_push_move(self):
        """Tests applying a move and checking the resulting state."""
        env = ChessEnv()
        move = chess.Move.from_uci("e2e4")
        state_planes, is_over, result = env.push_move(move)
        assert env.board.fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        assert state_planes.shape == (NUM_PLANES, 8, 8)
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
        """Tests the generation of legal moves."""
        env = ChessEnv()
        initial_legal_moves = env.get_legal_moves()
        assert len(initial_legal_moves) == 20

        env.push_move(chess.Move.from_uci("e2e4"))
        env.push_move(chess.Move.from_uci("e7e5"))
        legal_moves_after_e4e5 = env.get_legal_moves()
        assert len(legal_moves_after_e4e5) > 0

    def test_get_state_planes(self):
        """Tests the conversion of board state to neural network input planes."""
        env = ChessEnv()

        # Test piece positions (White Pawn at E2, then E4)
        initial_planes = env.get_state_planes()
        assert (
            initial_planes[
                chess.PAWN - 1 + WHITE_PIECE_PLANES_START, chess.square_rank(chess.E2), chess.square_file(chess.E2)
            ]
            == 1
        )

        env.push_move(chess.Move.from_uci("e2e4"))
        planes_after_e4 = env.get_state_planes()
        assert (
            planes_after_e4[
                chess.PAWN - 1 + WHITE_PIECE_PLANES_START, chess.square_rank(chess.E4), chess.square_file(chess.E4)
            ]
            == 1
        )
        assert (
            planes_after_e4[
                chess.PAWN - 1 + WHITE_PIECE_PLANES_START, chess.square_rank(chess.E2), chess.square_file(chess.E2)
            ]
            == 0
        )

        # Test black piece position (Black Pawn at E7, then E5)
        env.push_move(chess.Move.from_uci("e7e5"))
        planes_after_e5 = env.get_state_planes()
        assert (
            planes_after_e5[
                chess.PAWN - 1 + BLACK_PIECE_PLANES_START,
                chess.square_rank(chess.E5),
                chess.square_file(chess.E5),
            ]
            == 1
        )
        assert (
            planes_after_e5[
                chess.PAWN - 1 + BLACK_PIECE_PLANES_START,
                chess.square_rank(chess.E7),
                chess.square_file(chess.E7),
            ]
            == 0
        )

        # Test player to move (White then Black)
        assert initial_planes[WHITE_TO_MOVE_PLANE, 0, 0] == 1
        assert initial_planes[BLACK_TO_MOVE_PLANE, 0, 0] == 0
        assert planes_after_e4[WHITE_TO_MOVE_PLANE, 0, 0] == 0
        assert planes_after_e4[BLACK_TO_MOVE_PLANE, 0, 0] == 1

        # Test castling rights (initial)
        assert initial_planes[WHITE_KINGSIDE_CASTLING_PLANE, 0, 0] == 1
        assert initial_planes[WHITE_QUEENSIDE_CASTLING_PLANE, 0, 0] == 1
        assert initial_planes[BLACK_KINGSIDE_CASTLING_PLANE, 0, 0] == 1
        assert initial_planes[BLACK_QUEENSIDE_CASTLING_PLANE, 0, 0] == 1

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
        assert planes_ep[EN_PASSANT_PLANE, chess.square_rank(chess.C6), chess.square_file(chess.C6)] == 1

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
        assert planes_counters[FIFTY_MOVE_PLANE, 0, 0] == 2 / FIFTY_MOVE_NORMALIZATION_FACTOR

        # Fullmove number should be 3
        assert env.board.fullmove_number == 3
        assert planes_counters[FULLMOVE_PLANE, 0, 0] == (env.board.fullmove_number - 1) / FULLMOVE_NORMALIZATION_FACTOR

        # Test castling rights removal (White King moves)
        env.reset()
        env.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        env.push_move(chess.Move.from_uci("e1e2"))  # King moves, loses castling rights
        planes_castling_removed = env.get_state_planes()
        assert planes_castling_removed[WHITE_KINGSIDE_CASTLING_PLANE, 0, 0] == 0
        assert planes_castling_removed[WHITE_QUEENSIDE_CASTLING_PLANE, 0, 0] == 0
        assert planes_castling_removed[BLACK_KINGSIDE_CASTLING_PLANE, 0, 0] == 1
        assert planes_castling_removed[BLACK_QUEENSIDE_CASTLING_PLANE, 0, 0] == 1

        # Test en passant target square disappearance
        env.reset()
        env.push_move(chess.Move.from_uci("e2e4"))  # White pawn double push
        env.push_move(chess.Move.from_uci("a7a6"))  # Black makes a non-en-passant move
        planes_ep_disappear = env.get_state_planes()
        assert env.board.ep_square is None
        assert planes_ep_disappear[EN_PASSANT_PLANE, :, :].sum() == 0

        # Test promotion
        env.reset()
        env.board.set_fen("8/P7/8/8/8/8/8/K7 w - - 0 1")  # White pawn on a7
        env.push_move(chess.Move.from_uci("a7a8q"))  # Promote to Queen
        planes_promotion = env.get_state_planes()
        assert (
            planes_promotion[
                chess.PAWN - 1 + WHITE_PIECE_PLANES_START, chess.square_rank(chess.A7), chess.square_file(chess.A7)
            ]
            == 0
        )
        assert (
            planes_promotion[
                chess.QUEEN - 1 + WHITE_PIECE_PLANES_START,
                chess.square_rank(chess.A8),
                chess.square_file(chess.A8),
            ]
            == 1
        )

    def test_game_termination_conditions(self):
        """
        Tests various game termination conditions
        (stalemate, threefold repetition, fifty-move rule, insufficient material).
        """
        env = ChessEnv()

        # Test Stalemate
        env.board.set_fen("8/8/8/8/8/8/7k/7K w - - 0 1")  # Stalemate position
        assert env.is_game_over()
        assert env.result() == "1/2-1/2"

        # Test Threefold Repetition (using can_claim_threefold_repetition)
        env.reset()
        # Sequence: 1. Nf3 Nc6 2. Ng1 Nb8 3. Nf3 Nc6 4. Ng1 Nb8 5. Nf3
        # This sequence ensures the FEN (including halfmove clock, castling rights, en passant)
        # repeats exactly three times for the position after White's Nf3.
        moves_for_repetition = [
            "g1f3",
            "b8c6",
            "f3g1",
            "c6b8",
            "g1f3",
            "b8c6",
            "f3g1",
            "c6b8",
            "g1f3",
        ]
        for move_uci in moves_for_repetition:
            env.push_move(chess.Move.from_uci(move_uci))

        # At this point, the position after White's Nf3 has occurred 3 times.
        # The halfmove clock should be consistent as no captures or pawn moves occurred.
        assert env.board.can_claim_threefold_repetition()
        assert env.is_game_over()
        assert env.result() == "1/2-1/2"

        # Test Fifty-move Rule Draw
        env.reset()
        # Set a FEN where halfmove clock is high, and no pawn moves or captures occur
        # Example: 7k/8/8/8/8/8/8/7K w - - 99 50 (99 halfmoves, next move makes it 100)
        env.board.set_fen("7k/8/8/8/8/8/8/7K w - - 99 50")
        env.push_move(chess.Move.from_uci("h1g1"))  # Any non-pawn, non-capture move
        assert env.is_game_over()
        assert env.result() == "1/2-1/2"

        # Test Insufficient Material (King vs King)
        env.reset()
        env.board.set_fen("8/8/8/8/8/8/8/K6k w - - 0 1")
        assert env.is_game_over()
        assert env.result() == "1/2-1/2"

        # Test Insufficient Material (King + Bishop vs King)
        env.reset()
        env.board.set_fen("8/8/8/8/8/8/8/K1B4k w - - 0 1")
        assert env.is_game_over()
        assert env.result() == "1/2-1/2"
