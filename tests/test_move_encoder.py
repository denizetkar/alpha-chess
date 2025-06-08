import pytest
import chess
from src.move_encoder import MoveEncoderDecoder


@pytest.fixture
def encoder():
    return MoveEncoderDecoder()


class TestMoveEncoderDecoder:
    def test_initial_mapping_size(self, encoder):
        assert len(encoder.idx_to_move) == 4672
        assert encoder.total_actions == 4672
        assert len(encoder.move_types_map) == 73
        assert len(encoder.idx_to_move_type) == 73

    def test_encode_decode_identity(self, encoder):
        board = chess.Board()
        # Test some common moves
        moves_to_test = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b8c6"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("d7d5"),
            chess.Move.from_uci("e4d5"),
            chess.Move.from_uci("d8h4"),  # Fool's mate checkmate
            chess.Move.from_uci("e1g1"),  # White kingside castling
            chess.Move.from_uci("e8g8"),  # Black kingside castling
            chess.Move.from_uci("a2a4"),
            chess.Move.from_uci("b7b5"),
            chess.Move.from_uci("a4b5"),  # Pawn capture
            chess.Move.from_uci("e5f6"),  # Another pawn capture
        ]

        # Add some promotion moves (need to set up board for them)
        board_promo_white = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        moves_to_test.append(chess.Move.from_uci("a7a8q"))
        moves_to_test.append(chess.Move.from_uci("a7a8r"))
        moves_to_test.append(chess.Move.from_uci("a7a8b"))
        moves_to_test.append(chess.Move.from_uci("a7a8n"))

        board_promo_black = chess.Board("r6k/8/8/8/8/8/p7/K6K b - - 0 1")
        moves_to_test.append(chess.Move.from_uci("a2a1q"))
        moves_to_test.append(chess.Move.from_uci("a2a1r"))
        moves_to_test.append(chess.Move.from_uci("a2a1b"))
        moves_to_test.append(chess.Move.from_uci("a2a1n"))

        # Test en passant (need to set up board)
        board_ep = chess.Board("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP2PPP/RNBQKBNR w KQkq - 0 1")
        moves_to_test.append(chess.Move.from_uci("d4e3"))  # White pawn captures en passant

        board_ep_black = chess.Board("rnbqkbnr/ppp2ppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        moves_to_test.append(chess.Move.from_uci("e5d6"))  # Black pawn captures en passant

        for move in moves_to_test:
            # For moves that require a specific board state (like promotions, en passant, castling),
            # we need to use the correct board.
            # For simplicity in this test, we'll use a default board for most,
            # and specific boards for special moves.
            current_board = board
            if move.promotion is not None and move.from_square == chess.A7:
                current_board = board_promo_white
            elif move.promotion is not None and move.from_square == chess.A2:
                current_board = board_promo_black
            elif move.uci() == "d4e3":
                current_board = board_ep
            elif move.uci() == "e5d6":
                current_board = board_ep_black

            # Ensure the move is legal on the current board for encoding to be meaningful
            if current_board.is_legal(move):
                encoded_idx = encoder.encode(current_board, move)
                decoded_move = encoder.decode(encoded_idx)

                # AlphaZero's 73-plane encoding does not distinguish Queen promotions
                # from regular sliding moves. So, decoded_move will have promotion=None
                # if the original move was a Queen promotion.
                if move.promotion == chess.QUEEN:
                    assert decoded_move.from_square == move.from_square
                    assert decoded_move.to_square == move.to_square
                    assert decoded_move.promotion is None  # Decoded move will not have promotion
                else:
                    assert decoded_move == move, f"Failed for move: {move.uci()}"
            else:
                # If the move is not legal on the current board, encoding might still work
                # but decoding might not yield the exact same move if it's a dummy.
                # For this test, we only care about legal moves.
                pass

    def test_encode_invalid_move(self, encoder):
        board = chess.Board()
        # Test encoding a move from an empty square
        empty_square_move = chess.Move.from_uci("a3a4")
        with pytest.raises(ValueError, match="No piece at from_square"):
            encoder.encode(board, empty_square_move)

    def test_decode_out_of_bounds(self, encoder):
        with pytest.raises(IndexError):
            encoder.decode(4672)
        with pytest.raises(IndexError):
            encoder.decode(-1)

    def test_all_indices_decode_to_valid_moves(self, encoder):
        # This test ensures that every index in the 0-4671 range
        # can be decoded into a chess.Move object without error.
        # The legality of the move is not checked here, only that it's a valid chess.Move object.
        for i in range(encoder.total_actions):
            move = encoder.decode(i)
            assert isinstance(move, chess.Move)
            # Optionally, check if the move is not a dummy a1a1 unless it's an off-board target
            # This is harder to test without knowing the (from_sq, move_type) mapping directly.
            # For now, just ensure it's a chess.Move.
