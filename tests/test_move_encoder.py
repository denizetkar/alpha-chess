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

    @pytest.mark.parametrize(
        "fen, move_uci, is_promotion_to_queen",
        [
            # Common moves from initial board
            # Common moves from initial board
            (chess.STARTING_FEN, "e2e4", False),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "d7d5", False),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3", False),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "b8c6", False),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4", False),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "d7d5", False),
            ("rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "d7d5", False),  # After 1. e4
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "g1f3", False),  # After 1... e5
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "d1h5", False),  # Queen move
            (
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "d2d4",
                False,
            ),  # White kingside castling (replaced e1g1)
            (
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
                "b8c6",
                False,
            ),  # Black kingside castling (replaced e8g8)
            # Pawn captures
            ("rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "d7d5", False),  # 1. e4 d5
            ("rnbqkbnr/pppp1ppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e4d5", False),  # 1. e4 d5 2. exd5
            # Promotions
            ("8/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q", True),  # White promotion to Queen
            ("8/P7/8/8/8/8/8/7K w - - 0 1", "a7a8r", False),  # White promotion to Rook
            ("8/P7/8/8/8/8/8/7K w - - 0 1", "a7a8b", False),  # White promotion to Bishop
            ("8/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n", False),  # White promotion to Knight
            ("8/k7/8/8/8/8/p7/8 b - - 0 1", "a2a1q", True),  # Black promotion to Queen
            ("8/k7/8/8/8/8/p7/8 b - - 0 1", "a2a1r", False),  # Black promotion to Rook
            ("8/k7/8/8/8/8/p7/8 b - - 0 1", "a2a1b", False),  # Black promotion to Bishop
            ("8/k7/8/8/8/8/p7/8 b - - 0 1", "a2a1n", False),  # Black promotion to Knight
            # En passant
            ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6", False),  # White en passant
            ("rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3", "d4e3", False),  # Black en passant
            # Castling moves
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", False),  # White kingside castling
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", False),  # White queenside castling
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", False),  # Black kingside castling
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", False),  # Black queenside castling
        ],
    )
    def test_encode_decode_identity(self, encoder, fen, move_uci, is_promotion_to_queen):
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)

        # Ensure the move is legal on the current board for encoding to be meaningful
        assert board.is_legal(move), f"Move {move_uci} is not legal on board {fen}"

        encoded_idx = encoder.encode(board, move)
        decoded_move = encoder.decode(encoded_idx)

        # AlphaZero's 73-plane encoding does not distinguish Queen promotions
        # from regular sliding moves. So, decoded_move will have promotion=None
        # if the original move was a Queen promotion.
        if is_promotion_to_queen:
            assert decoded_move.from_square == move.from_square
            assert decoded_move.to_square == move.to_square
            assert decoded_move.promotion is None  # Decoded move will not have promotion
        else:
            assert decoded_move == move, f"Failed for move: {move.uci()} on board {fen}"

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
