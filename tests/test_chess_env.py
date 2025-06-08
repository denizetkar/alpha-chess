import chess
from src.chess_env import ChessEnv


class TestChessEnv:
    def test_initial_state(self):
        env = ChessEnv()
        initial_planes = env.reset()
        assert initial_planes.shape == (29, 8, 8)  # Updated to 29 planes
        assert env.board.fen() == chess.STARTING_FEN
        assert not env.is_game_over()

    def test_reset_history(self):
        env = ChessEnv()
        # Push some moves to build up history
        env.push_move(chess.Move.from_uci("e2e4"))
        env.push_move(chess.Move.from_uci("e7e5"))
        env.push_move(chess.Move.from_uci("g1f3"))
        assert len(env.board_history) == 3

        # Reset the environment
        env.reset()

        # Verify history is cleared and then contains only the initial board
        assert len(env.board_history) == 1  # After reset, _update_history is called once
        assert env.board_history[0].fen() == chess.STARTING_FEN
        assert env.board.fen() == chess.STARTING_FEN

    def test_board_history_management(self):
        env = ChessEnv()
        # Push more than 8 moves to test maxlen
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d4", "e5d4", "c3d4"]
        for move_uci in moves:
            env.push_move(chess.Move.from_uci(move_uci))

        # Verify maxlen=8
        assert len(env.board_history) == 8

        # Verify that board.copy() is used (immutability)
        # Get a board from history
        historical_board = env.board_history[0]
        # Make a change to the current board
        env.board.push(chess.Move.from_uci("d4e5"))
        # Ensure the historical board remains unchanged
        assert historical_board.fen() != env.board.fen()

        # Verify history planes (21-28)
        # After 11 moves, the history should contain boards from move 4 to move 11 (inclusive)
        # The current board is after move 11 (c3d4)
        # History: [board after e7e5, board after g1f3, ..., board after c3d4]
        # The last board in history is the current board before the last push_move.
        # The history planes are 21-28, representing the current board + 7 previous.
        # The `_update_history` is called *after* the move is pushed.
        # So, after 11 moves, the history contains 11 boards.
        # The deque's maxlen=8 means it contains the last 8 boards.
        # Let's reset and build history carefully to test planes.

        env.reset()
        test_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"]
        for move_uci in test_moves:
            env.push_move(chess.Move.from_uci(move_uci))

        # Now history should have 8 boards.
        # The current board is after "g8f6" (Black's turn)
        # The history planes should reflect the turn of the player for each historical board.
        # Plane 21: Oldest history (board after e2e4, Black to move)
        # Plane 28: Current board (board after g8f6, White to move for next move)

        planes = env.get_state_planes()

        # Check the turn for each historical board in the planes
        # The board history stores the board *after* the move.
        # So, board_history[0] is board after e2e4 (Black to move)
        # board_history[1] is board after e7e5 (White to move)
        # ...
        # board_history[7] is board after g8f6 (White to move)

        # The planes are indexed 21-28.
        # planes[21] corresponds to board_history[0]
        # planes[22] corresponds to board_history[1]
        # ...
        # planes[28] corresponds to board_history[7]

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
                assert (planes[21 + i, :, :] == 1).all(), f"Plane {21+i} should be all ones (White to move)"
            else:
                assert (planes[21 + i, :, :] == 0).all(), f"Plane {21+i} should be all zeros (Black to move)"

    def test_push_move(self):
        env = ChessEnv()
        move = chess.Move.from_uci("e2e4")
        state_planes, is_over, result = env.push_move(move)
        assert env.board.fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        assert state_planes.shape == (29, 8, 8)  # Updated to 29 planes
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

        # Test castling rights removal (White King moves)
        env.reset()
        env.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        env.push_move(chess.Move.from_uci("e1e2"))  # King moves, loses castling rights
        planes_castling_removed = env.get_state_planes()
        assert planes_castling_removed[14, 0, 0] == 0  # White King-side lost
        assert planes_castling_removed[15, 0, 0] == 0  # White Queen-side lost
        assert planes_castling_removed[16, 0, 0] == 1  # Black King-side still present
        assert planes_castling_removed[17, 0, 0] == 1  # Black Queen-side still present

        # Test en passant target square disappearance
        env.reset()
        env.push_move(chess.Move.from_uci("e2e4"))  # White pawn double push
        env.push_move(chess.Move.from_uci("a7a6"))  # Black makes a non-en-passant move
        planes_ep_disappear = env.get_state_planes()
        assert env.board.ep_square is None
        assert planes_ep_disappear[18, :, :].sum() == 0  # En passant plane should be all zeros

        # Test promotion
        env.reset()
        env.board.set_fen("8/P7/8/8/8/8/8/K7 w - - 0 1")  # White pawn on a7
        env.push_move(chess.Move.from_uci("a7a8q"))  # Promote to Queen
        planes_promotion = env.get_state_planes()
        assert planes_promotion[0, chess.square_rank(chess.A7), chess.square_file(chess.A7)] == 0  # Old pawn gone
        assert (
            planes_promotion[4, chess.square_rank(chess.A8), chess.square_file(chess.A8)] == 1
        )  # New white queen at A8

    def test_game_termination_conditions(self):
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
            "b8c6",  # Pos A (1st), Pos B (1st)
            "f3g1",
            "c6b8",  # Pos C (1st), Pos D (1st)
            "g1f3",
            "b8c6",  # Pos A (2nd), Pos B (2nd)
            "f3g1",
            "c6b8",  # Pos C (2nd), Pos D (2nd)
            "g1f3",  # Pos A (3rd)
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
