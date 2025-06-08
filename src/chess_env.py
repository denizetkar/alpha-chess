import chess
import numpy as np
from collections import deque

# Constants for plane indices
WHITE_PIECE_PLANES_START = 0
BLACK_PIECE_PLANES_START = 6
WHITE_TO_MOVE_PLANE = 12
BLACK_TO_MOVE_PLANE = 13
WHITE_KINGSIDE_CASTLING_PLANE = 14
WHITE_QUEENSIDE_CASTLING_PLANE = 15
BLACK_KINGSIDE_CASTLING_PLANE = 16
BLACK_QUEENSIDE_CASTLING_PLANE = 17
EN_PASSANT_PLANE = 18
FIFTY_MOVE_PLANE = 19
FULLMOVE_PLANE = 20
HISTORY_PLANES_START = 21
NUM_HISTORY_PLANES = 8
NUM_PLANES = 29

# Normalization constants
FIFTY_MOVE_NORMALIZATION_FACTOR = 100.0
FULLMOVE_NORMALIZATION_FACTOR = 2000.0


class ChessEnv:
    """
    Represents the chess environment, handling board state, move generation,
    and conversion to neural network input planes.
    """

    def __init__(self) -> None:
        self.board: chess.Board = chess.Board()
        self.board_history: deque[chess.Board] = deque(
            maxlen=NUM_HISTORY_PLANES
        )  # Store last 8 half-moves for input planes

    def reset(self) -> np.ndarray:
        """Resets the board to the initial state."""
        self.board.reset()
        self.board_history.clear()
        self._update_history()
        return self.get_state_planes()

    def push_move(self, move: chess.Move) -> tuple[np.ndarray, bool, str]:
        """Applies a move to the board."""
        self.board.push(move)
        self._update_history()
        return self.get_state_planes(), self.is_game_over(), self.result()

    def _update_history(self) -> None:
        """Adds the current board state to history."""
        # Store a copy of the board to prevent modification issues
        self.board_history.append(self.board.copy())

    def get_legal_moves(self) -> list[chess.Move]:
        """Returns a list of legal moves."""
        return list(self.board.legal_moves)

    def get_state_planes(self) -> np.ndarray:
        """
        Converts the current board state into a 3D numpy array of binary planes
        suitable for neural network input.

        The planes are ordered as follows:
        - White pieces (P, N, B, R, Q, K): Planes 0-5
        - Black pieces (P, N, B, R, Q, K): Planes 6-11
        - Player to move (White/Black): Planes 12-13
        - Castling rights (White/Black King-side/Queen-side): Planes 14-17
        - En passant target square: Plane 18
        - Fifty-move rule counter (normalized): Plane 19
        - Fullmove number (normalized): Plane 20
        - Last 8 half-moves (current board + 7 previous boards): Planes 21-28
          Each of these 8 planes indicates the player to move for that historical board (1 for White, 0 for Black).

        Total planes: 29 (8x8 each).
        """
        # Initialize with 29 planes
        planes: np.ndarray = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

        # Piece planes (0-11)
        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            row: int = chess.square_rank(square)
            col: int = chess.square_file(square)
            if piece.color == chess.WHITE:
                planes[piece.piece_type - 1 + WHITE_PIECE_PLANES_START, row, col] = 1
            else:
                planes[piece.piece_type - 1 + BLACK_PIECE_PLANES_START, row, col] = 1

        # Player to move (12-13)
        if self.board.turn == chess.WHITE:
            planes[WHITE_TO_MOVE_PLANE, :, :] = 1
        else:
            planes[BLACK_TO_MOVE_PLANE, :, :] = 1

        # Castling rights (14-17)
        if self.board.has_kingside_castling_rights(chess.WHITE):
            planes[WHITE_KINGSIDE_CASTLING_PLANE, :, :] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            planes[WHITE_QUEENSIDE_CASTLING_PLANE, :, :] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            planes[BLACK_KINGSIDE_CASTLING_PLANE, :, :] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            planes[BLACK_QUEENSIDE_CASTLING_PLANE, :, :] = 1

        # En passant target square (18)
        if self.board.ep_square:
            row, col = chess.square_rank(self.board.ep_square), chess.square_file(self.board.ep_square)
            planes[EN_PASSANT_PLANE, row, col] = 1

        # Fifty-move rule counter (19) - normalized
        # Max halfmove_clock for fifty-move rule is 100.
        planes[FIFTY_MOVE_PLANE, :, :] = self.board.halfmove_clock / FIFTY_MOVE_NORMALIZATION_FACTOR

        # Fullmove number (20) - normalized
        # Fullmove number can go very high, normalize by a large constant.
        # A typical game might have 50-100 full moves, but can be much more.
        # Using 2000 as a rough upper bound for normalization.
        planes[FULLMOVE_PLANE, :, :] = (self.board.fullmove_number - 1) / FULLMOVE_NORMALIZATION_FACTOR

        # Move history planes (21-28)
        # Iterate through history from oldest to newest, filling planes from 21 onwards
        for i, hist_board in enumerate(self.board_history):
            if i < NUM_HISTORY_PLANES:  # Ensure we don't go out of bounds for the 8 history planes
                if hist_board.turn == chess.WHITE:
                    planes[HISTORY_PLANES_START + i, :, :] = 1  # White to move in this historical state
                # Else, it's 0 (already initialized) for Black to move

        return planes

    def is_game_over(self) -> bool:
        """
        Checks if the game is over, including claimable draws like threefold repetition.
        """
        if self.board.is_game_over():
            return True
        # Explicitly check for claimable threefold repetition
        if self.board.can_claim_threefold_repetition():
            return True
        return False

    def result(self) -> str:
        """
        Returns the game result string.
        If the game is over by threefold repetition, returns '1/2-1/2'.
        """
        if self.board.can_claim_threefold_repetition():
            return "1/2-1/2"
        return self.board.result()

    def __str__(self) -> str:
        """String representation of the board."""
        return str(self.board)
