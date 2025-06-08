import chess
from typing import List, Dict, Optional, Tuple

MOVE_ENCODING_SIZE = 64 * 73  # 4672 total possible moves


class MoveEncoderDecoder:
    """
    Encodes and decodes chess moves into a fixed-size integer index,
    following an AlphaZero-like 73-plane encoding scheme.

    The action space is 64 squares * 73 move types = 4672 actions.
    Each move is represented by (from_square, move_type_idx).
    """

    def __init__(self):
        self.move_to_idx: Dict[chess.Move, int] = {}  # Populated dynamically in encode
        self.idx_to_move: List[Optional[chess.Move]] = [None] * MOVE_ENCODING_SIZE
        self.total_actions = MOVE_ENCODING_SIZE

        # Define the 73 move types (delta_row, delta_col, promotion_piece_type)
        self.move_types_map: Dict[Tuple[int, int, Optional[int]], int] = {}
        self.idx_to_move_type: List[Tuple[int, int, Optional[int]]] = []

        sliding_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        # 1. Single-step sliding moves (for King and step-1 Queen/Rook/Bishop) - 8 types
        for dr, dc in sliding_directions:
            self.idx_to_move_type.append((dr, dc, None))  # step = 1

        # 2. Multi-step sliding moves (for Queen, Rook, Bishop) - 48 types (8 directions * 6 steps)
        for dr, dc in sliding_directions:
            for step in range(2, 8):  # Start from step 2
                self.idx_to_move_type.append((dr * step, dc * step, None))

        # 3. Knight moves - 8 types
        knight_deltas = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        for dr, dc in knight_deltas:
            self.idx_to_move_type.append((dr, dc, None))

        # 4. Underpromotions - 9 types
        promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        promotion_deltas = [(1, 0), (1, -1), (1, 1)]  # Straight, diag-left, diag-right (for white)

        for dr, dc in promotion_deltas:
            for piece_type in promotion_pieces:
                self.idx_to_move_type.append((dr, dc, piece_type))

        # Populate move_types_map
        for i, move_type in enumerate(self.idx_to_move_type):
            self.move_types_map[move_type] = i

        assert len(self.idx_to_move_type) == 73, f"Expected 73 move types, got {len(self.idx_to_move_type)}"

        # Populate idx_to_move with all 4672 possible moves (from_square, move_type)
        for from_sq in chess.SQUARES:
            for move_type_idx, (dr, dc, promo_piece_type) in enumerate(self.idx_to_move_type):
                from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
                # Adjust dr for black pawn promotions
                adjusted_dr = dr
                if promo_piece_type is not None:
                    # If it's a promotion move type, and the from_sq is on the 2nd rank (for black pawn)
                    # then the pawn is black and moves 'up' the board (rank 1 to 0), so dr should be -1.
                    # The promo_deltas are defined as (1,0), (1,-1), (1,1) for white's perspective.
                    # So if from_rank is 1 (for black pawn on 2nd rank), dr should be -1.
                    if from_rank == 1:  # This is a black pawn on its 2nd rank
                        adjusted_dr = -dr  # Flip the delta for black pawns
                    # No need to check for white pawn on 7th rank (rank 6) as dr=1 is already correct.

                to_rank, to_file = from_rank + adjusted_dr, from_file + dc

                # Check if to_square is on board
                if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                    # This move type from this square leads off-board.
                    # We still need to assign an index, but the move itself would be illegal.
                    # For these cases, we create a dummy move that is always illegal (e.g., a1a1).
                    # The NN will still output a probability for this index, but it will be masked later.
                    move = chess.Move(chess.A1, chess.A1)
                else:
                    to_sq = chess.square(to_file, to_rank)
                    move = chess.Move(from_sq, to_sq, promotion=promo_piece_type)

                global_idx = from_sq * 73 + move_type_idx
                if global_idx >= self.total_actions:
                    raise ValueError("Index out of bounds during mapping initialization.")
                self.idx_to_move[global_idx] = move

    def encode(self, board: chess.Board, move: chess.Move) -> int:
        """
        Encodes a chess.Move object into a unique integer index (0-4671)
        based on the AlphaZero 73-plane encoding scheme.
        Requires the current board state to determine the piece type.
        """
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion

        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
        to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)

        dr, dc = to_rank - from_rank, to_file - from_file

        piece = board.piece_at(from_sq)
        if piece is None:
            raise ValueError(f"No piece at from_square {chess.square_name(from_sq)} for move {move.uci()}")

        move_type_idx = -1

        # Handle promotions first (pawn moves to 8th/1st rank with promotion)
        if promotion is not None:
            # Determine the relative delta for the pawn move based on color
            if piece.color == chess.WHITE:
                promo_dr = dr
                promo_dc = dc
            else:  # Black
                promo_dr = -dr
                promo_dc = -dc

            promo_move_type_tuple = (promo_dr, promo_dc, promotion)
            if promo_move_type_tuple in self.move_types_map:
                move_type_idx = self.move_types_map[promo_move_type_tuple]
            else:
                # If it's a promotion to Queen, it should be handled as a sliding move.
                # Otherwise, it's an invalid underpromotion.
                if promotion == chess.QUEEN:
                    # Fall through to the sliding move logic below
                    pass
                else:
                    raise ValueError(f"Promotion move {move.uci()} not found in defined underpromotion types.")

        # Handle non-promotion moves and Queen promotions
        if move_type_idx == -1:  # Only proceed if move_type_idx hasn't been found yet (i.e., not an underpromotion)
            if board.is_castling(move):
                # Castling moves are special King moves that cover 2 squares.
                # They should be mapped to the corresponding 2-step sliding move type.
                # The dr, dc are already calculated.
                step = max(abs(dr), abs(dc))
                norm_dr = dr // step if step != 0 else 0
                norm_dc = dc // step if step != 0 else 0

                castling_move_type_tuple = (norm_dr * step, norm_dc * step, None)
                if castling_move_type_tuple in self.move_types_map:
                    move_type_idx = self.move_types_map[castling_move_type_tuple]
                else:
                    raise ValueError(f"Castling move {move.uci()} not found in defined sliding types.")
            elif piece.piece_type == chess.KNIGHT:
                # Knight moves
                # Find index in knight_deltas (indices 56-63)
                for i, (kdr, kdc, _) in enumerate(self.idx_to_move_type[56:64]):
                    if dr == kdr and dc == kdc:
                        move_type_idx = 56 + i
                        break
                if move_type_idx == -1:
                    raise ValueError(f"Knight move {move.uci()} not found in defined knight moves.")
            elif piece.piece_type == chess.KING:
                king_move_tuple = (dr, dc, None)
                if king_move_tuple in self.move_types_map:
                    idx = self.move_types_map[king_move_tuple]
                    if 0 <= idx < 8:  # Ensure it's one of the 1-step sliding moves
                        move_type_idx = idx
                    else:
                        raise ValueError(f"King move {move.uci()} mapped to incorrect move type index {idx}.")
                else:
                    raise ValueError(f"King move {move.uci()} not found in defined single-step moves.")
            else:  # Queen, Rook, Bishop, Pawn (non-promotion), or Queen promotion
                # Sliding moves (including pawn pushes/captures that fit a sliding pattern)
                # Determine step for sliding moves
                if dr == 0 and dc == 0:  # Should not happen for a valid move
                    raise ValueError(f"Invalid move delta (0,0) for move {move.uci()}")

                # Calculate step for sliding moves
                step = 0
                if dr != 0:
                    step = abs(dr)
                elif dc != 0:
                    step = abs(dc)

                # Normalize dr, dc to get direction
                norm_dr = dr // step if step != 0 else 0
                norm_dc = dc // step if step != 0 else 0

                sliding_move_type_tuple = (norm_dr * step, norm_dc * step, None)
                if sliding_move_type_tuple in self.move_types_map:
                    move_type_idx = self.move_types_map[sliding_move_type_tuple]
                else:
                    raise ValueError(f"Sliding move {move.uci()} not found in defined sliding types.")

        if move_type_idx == -1:
            raise ValueError(f"Could not determine move type for move: {move.uci()}")

        global_idx = from_sq * 73 + move_type_idx
        if global_idx >= self.total_actions:
            raise IndexError(
                f"Calculated global index {global_idx} out of bounds for total actions {self.total_actions}"
            )

        # Store the mapping for future lookups (memoization)
        self.move_to_idx[move] = global_idx
        return global_idx

    def decode(self, idx: int) -> chess.Move:
        """
        Decodes an integer index (0-4671) back into a chess.Move object.
        """
        if not (0 <= idx < self.total_actions):
            raise IndexError(f"Index {idx} out of bounds for move decoding. Total actions: {self.total_actions}")

        # Extract from_square and move_type_idx
        from_sq = idx // 73
        move_type_idx = idx % 73

        dr, dc, promo_piece_type = self.idx_to_move_type[move_type_idx]

        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)

        # Adjust dr for black pawn promotions in decode method
        adjusted_dr = dr
        if promo_piece_type is not None:
            # If it's a promotion move type, and the from_sq is on the 2nd rank (for black pawn)
            if from_rank == 1:  # This is a black pawn on its 2nd rank
                adjusted_dr = -dr  # Flip the delta for black pawns

        to_rank, to_file = from_rank + adjusted_dr, from_file + dc

        # Check if to_square is on board
        if not (0 <= to_rank < 8 and 0 <= to_file < 8):
            # This index corresponds to an off-board target.
            # Return a dummy illegal move, as this index should be masked by legal moves later.
            return chess.Move(chess.A1, chess.A1)  # Or raise an error if strict legality is needed here

        to_sq = chess.square(to_file, to_rank)
        return chess.Move(from_sq, to_sq, promotion=promo_piece_type)
