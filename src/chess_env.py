import chess
import numpy as np
from collections import deque


class ChessEnv:
    """
    Represents the chess environment, handling board state, move generation,
    and conversion to neural network input planes.
    """

    def __init__(self):
        self.board = chess.Board()
        self.board_history = deque(maxlen=8)  # Store last 8 half-moves for input planes

    def reset(self):
        """Resets the board to the initial state."""
        self.board.reset()
        self.board_history.clear()
        self._update_history()
        return self.get_state_planes()

    def push_move(self, move: chess.Move):
        """Applies a move to the board."""
        self.board.push(move)
        self._update_history()
        return self.get_state_planes(), self.board.is_game_over(), self.board.result()

    def _update_history(self):
        """Adds the current board state to history."""
        # Store a copy of the board to prevent modification issues
        self.board_history.append(self.board.copy())

    def get_legal_moves(self):
        """Returns a list of legal moves."""
        return list(self.board.legal_moves)

    def get_state_planes(self) -> np.ndarray:
        """
        Converts the current board state into a 3D numpy array of binary planes
        suitable for neural network input.

        The planes are ordered as follows:
        0-5: White pieces (P, N, B, R, Q, K)
        6-11: Black pieces (P, N, B, R, Q, K)
        12: All ones if white to move, all zeros if black to move
        13: All ones if black to move, all zeros if white to move
        14: Castling rights (White King-side)
        15: Castling rights (White Queen-side)
        16: Castling rights (Black King-side)
        17: Castling rights (Black Queen-side)
        18: En passant target square
        19: Fifty-move rule counter (normalized)
        20: Fullmove number (normalized)
        21-28: Last 8 half-moves (current board + 7 previous boards)
               Each of these 8 planes indicates the player to move for that historical board.
               1 if white to move, 0 if black to move.

        Total planes: 12 + 2 + 4 + 1 + 1 + 1 + 8 = 29 planes.
        Each plane is 8x8.
        """
        # Initialize with 29 planes
        planes = np.zeros((29, 8, 8), dtype=np.float32)

        # Piece planes (0-11)
        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            row, col = chess.square_rank(square), chess.square_file(square)
            if piece.color == chess.WHITE:
                planes[piece.piece_type - 1, row, col] = 1
            else:
                planes[piece.piece_type - 1 + 6, row, col] = 1

        # Player to move (12-13)
        if self.board.turn == chess.WHITE:
            planes[12, :, :] = 1
        else:
            planes[13, :, :] = 1

        # Castling rights (14-17)
        if self.board.has_kingside_castling_rights(chess.WHITE):
            planes[14, :, :] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            planes[15, :, :] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            planes[16, :, :] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            planes[17, :, :] = 1

        # En passant target square (18)
        if self.board.ep_square:
            row, col = chess.square_rank(self.board.ep_square), chess.square_file(self.board.ep_square)
            planes[18, row, col] = 1

        # Fifty-move rule counter (19) - normalized
        # Max halfmove_clock for fifty-move rule is 100.
        planes[19, :, :] = self.board.halfmove_clock / 100.0

        # Fullmove number (20) - normalized
        # Fullmove number can go very high, normalize by a large constant.
        # A typical game might have 50-100 full moves, but can be much more.
        # Using 2000 as a rough upper bound for normalization.
        planes[20, :, :] = (self.board.fullmove_number - 1) / 2000.0

        # Move history planes (21-28)
        # Iterate through history from oldest to newest, filling planes from 21 onwards
        for i, hist_board in enumerate(self.board_history):
            if i < 8:  # Ensure we don't go out of bounds for the 8 history planes
                if hist_board.turn == chess.WHITE:
                    planes[21 + i, :, :] = 1  # White to move in this historical state
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

    def __str__(self):
        """String representation of the board."""
        return str(self.board)


# Example Usage (for testing)
if __name__ == "__main__":
    env = ChessEnv()
    print("Initial Board:")
    print(env)
    print(f"Is game over: {env.is_game_over()}")
    print(f"Legal moves: {len(env.get_legal_moves())}")

    # Test move
    move = chess.Move.from_uci("e2e4")
    state_planes, is_over, result = env.push_move(move)
    print("\nBoard after e2e4:")
    print(env)
    print(f"Is game over: {is_over}")
    print(f"Result: {result}")
    print(f"State planes shape: {state_planes.shape}")
    white_pawn_e4_rank = chess.square_rank(chess.E4)
    white_pawn_e4_file = chess.square_file(chess.E4)
    print(
        f"Value at planes[0, {white_pawn_e4_rank}, {white_pawn_e4_file}] (white pawn at e4): "
        f"{state_planes[0, white_pawn_e4_rank, white_pawn_e4_file]}"
    )  # Should be 1

    # Test another move
    move = chess.Move.from_uci("e7e5")
    state_planes, is_over, result = env.push_move(move)
    print("\nBoard after e7e5:")
    print(env)
    print(f"Is game over: {is_over}")
    print(f"Result: {result}")
    print(f"State planes shape: {state_planes.shape}")
    black_pawn_e5_rank = chess.square_rank(chess.E5)
    black_pawn_e5_file = chess.square_file(chess.E5)
    print(
        f"Value at planes[6, {black_pawn_e5_rank}, {black_pawn_e5_file}] (black pawn at e5): "
        f"{state_planes[6, black_pawn_e5_rank, black_pawn_e5_file]}"
    )  # Should be 1

    # Test castling rights
    env.reset()
    print("\nInitial castling rights planes:")
    print(f"White King-side: {env.get_state_planes()[14, 0, 0]}")
    print(f"White Queen-side: {env.get_state_planes()[15, 0, 0]}")
    print(f"Black King-side: {env.get_state_planes()[16, 0, 0]}")
    print(f"Black Queen-side: {env.get_state_planes()[17, 0, 0]}")

    # Test en passant
    env.reset()
    env.push_move(chess.Move.from_uci("e2e4"))
    env.push_move(chess.Move.from_uci("d7d5"))
    env.push_move(chess.Move.from_uci("e4d5"))
    env.push_move(chess.Move.from_uci("g8f6"))
    env.push_move(chess.Move.from_uci("d5d4"))
    env.push_move(chess.Move.from_uci("c7c5"))
    print("\nBoard before en passant:")
    print(env)
    print(f"En passant square: {env.board.ep_square}")
    if env.board.ep_square is not None:
        ep_square_rank = chess.square_rank(env.board.ep_square)
        ep_square_file = chess.square_file(env.board.ep_square)
        print(f"En passant plane value: {env.get_state_planes()[18, ep_square_rank, ep_square_file]}")
    else:
        print("En passant square is None, no en passant plane value to check.")

    # Test game over (checkmate example - Fool's Mate)
    # Create a new environment for this test to ensure clean state
    checkmate_env = ChessEnv()
    print("\nStarting Fool's Mate sequence:")
    print(checkmate_env)

    # 1. f3
    checkmate_env.push_move(chess.Move.from_uci("f2f3"))
    print("\nAfter 1. f3:")
    print(checkmate_env)

    # 1... e5
    checkmate_env.push_move(chess.Move.from_uci("e7e5"))
    print("\nAfter 1... e5:")
    print(checkmate_env)

    # 2. g4
    checkmate_env.push_move(chess.Move.from_uci("g2g4"))
    print("\nAfter 2. g4:")
    print(checkmate_env)

    # 2... Qh4#
    checkmate_move = chess.Move.from_uci("d8h4")  # Black Queen from d8 to h4
    state_planes, is_over, result = checkmate_env.push_move(checkmate_move)
    print("\nBoard after checkmate (Qh4#):")
    print(f"Is game over: {is_over}")
    print(f"Result: {result}")
    print(f"Is checkmate: {is_over and checkmate_env.board.is_checkmate()}")  # Checkmate implies game over
    print(f"Is in check: {checkmate_env.board.is_check()}")
    print(f"Is stalemate: {checkmate_env.board.is_stalemate()}")
