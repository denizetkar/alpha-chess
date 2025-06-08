import torch
import chess
import yaml
import os
from tqdm import tqdm  # Import tqdm
from src.chess_env import ChessEnv
from src.nn_model import AlphaChessNet
from src.mcts import MCTSNode, MCTS
from src.move_encoder import MoveEncoderDecoder


class Tester:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.chess_env = ChessEnv()
        self.move_encoder = MoveEncoderDecoder()
        self.model: torch.nn.Module = AlphaChessNet(  # Explicitly type as torch.nn.Module
            num_residual_blocks=self.config["model"]["num_residual_blocks"],
            num_filters=self.config["model"]["num_filters"],
        ).to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Compile the model if PyTorch 2.0+ is available and configured
        if self.config["testing"]["use_torch_compile"] and hasattr(torch, "compile"):
            print("Compiling model with torch.compile for testing...")
            self.model = torch.compile(self.model)  # type: ignore

        self.checkpoint_dir = self.config["checkpointing"]["checkpoint_dir"]

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore
        print(f"Model loaded from {checkpoint_path}")

    def _play_game_with_agent(self, agent_color: chess.Color):
        self.chess_env.reset()
        current_board = self.chess_env.board.copy()
        mcts = MCTS(
            self.model,
            self.chess_env,
            self.move_encoder,
            c_puct=self.config["mcts"]["c_puct"],
        )  # type: ignore

        game_moves = []
        while not current_board.is_game_over():
            if current_board.turn == agent_color:
                # Agent's turn
                root_node = MCTSNode(current_board)
                mcts.run_simulations(root_node, self.config["mcts"]["simulations_per_move"])

                # Select move based on visit counts (deterministic for testing)
                if root_node.N > 0:
                    best_move = None
                    max_visits = -1
                    for move, child_node in root_node.children.items():
                        if child_node.N > max_visits:
                            max_visits = child_node.N
                            best_move = move
                    chosen_move = best_move
                else:
                    # Fallback if MCTS somehow fails (e.g., no simulations run)
                    legal_moves_list = list(current_board.legal_moves)
                    chosen_move = legal_moves_list[0] if legal_moves_list else None
            else:
                # Human's turn (or other agent's turn in future)
                # For now, this is a placeholder for human input in CLI mode.
                # In self-play mode, this branch won't be taken.
                chosen_move = None  # Will be handled by CLI input loop

            if chosen_move is None:
                # This can happen if no legal moves or MCTS fails to find one.
                # Or if it's human's turn and no input yet.
                break

            game_moves.append(chosen_move)
            current_board.push(chosen_move)
            self.chess_env.board = current_board.copy()  # Update env board

        return current_board.result(), game_moves

    def run_self_play_test(self, num_games: int):
        print(f"\n--- Running {num_games} Self-Play Test Games ---")
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        for i in tqdm(range(num_games), desc="Self-Play Test"):
            # Agent plays against itself (White vs Black)
            # For simplicity, let's assume the agent always plays both sides.
            # The MCTS will be initialized with the current model.

            # The _play_game_with_agent method is designed for agent vs human.
            # For self-play, we just need to run the self-play loop from train.py
            # but without data collection or training.

            self.chess_env.reset()
            current_board = self.chess_env.board.copy()
            root_node = MCTSNode(current_board)
            mcts = MCTS(
                self.model,
                self.chess_env,
                self.move_encoder,
                c_puct=self.config["mcts"]["c_puct"],
            )  # type: ignore

            while not current_board.is_game_over():
                mcts.run_simulations(root_node, self.config["mcts"]["simulations_per_move"])

                if root_node.N > 0:
                    # Select move based on visit counts (deterministic for testing)
                    best_move = None
                    max_visits = -1
                    for move, child_node in root_node.children.items():
                        if child_node.N > max_visits:
                            max_visits = child_node.N
                            best_move = move
                    chosen_move = best_move
                else:
                    legal_moves_list = list(current_board.legal_moves)
                    chosen_move = legal_moves_list[0] if legal_moves_list else None

                if chosen_move is None:
                    break

                current_board.push(chosen_move)
                self.chess_env.board = current_board.copy()

                if chosen_move in root_node.children:
                    root_node = root_node.children[chosen_move]
                    root_node.parent = None
                else:
                    root_node = MCTSNode(current_board)

            game_result = current_board.result()
            results[game_result] += 1

        print("\nSelf-Play Test Results:")
        for res, count in results.items():
            print(f"{res}: {count} games")
        print(f"Total games: {num_games}")

    def run_human_play_test(self):
        print("\n--- Starting Human vs AlphaChess (CLI) ---")
        print("Enter moves in UCI format (e.g., 'e2e4'). Type 'quit' to exit.")

        self.chess_env.reset()
        current_board = self.chess_env.board.copy()
        mcts = MCTS(
            self.model,
            self.chess_env,
            self.move_encoder,
            c_puct=self.config["mcts"]["c_puct"],
        )  # type: ignore

        while not current_board.is_game_over():
            print("\n" + str(current_board))
            print(f"Turn: {'White' if current_board.turn == chess.WHITE else 'Black'}")

            if current_board.turn == chess.WHITE:  # Assuming human plays White
                try:
                    uci_move = input("Your move (UCI): ")
                    if uci_move.lower() == "quit":
                        print("Exiting game.")
                        break

                    move = chess.Move.from_uci(uci_move)
                    if move not in current_board.legal_moves:
                        print("Illegal move. Try again.")
                        continue
                    current_board.push(move)
                except ValueError:
                    print("Invalid UCI format. Try again.")
                    continue
            else:  # Agent plays Black
                print("AlphaChess is thinking...")
                root_node = MCTSNode(current_board)
                mcts.run_simulations(root_node, self.config["mcts"]["simulations_per_move"])

                if root_node.N > 0:
                    best_move = None
                    max_visits = -1
                    for move, child_node in root_node.children.items():
                        if child_node.N > max_visits:
                            max_visits = child_node.N
                            best_move = move
                    chosen_move = best_move
                else:
                    legal_moves_list = list(current_board.legal_moves)
                    chosen_move = legal_moves_list[0] if legal_moves_list else None

                if chosen_move is None:
                    print("AlphaChess could not find a move. Game over.")
                    break

                print(f"AlphaChess plays: {chosen_move.uci()}")
                current_board.push(chosen_move)

            self.chess_env.board = current_board.copy()  # Update env board

        print("\nGame Over!")
        print(f"Result: {current_board.result()}")


if __name__ == "__main__":
    # Create a dummy config.yaml if it doesn't exist
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        dummy_config_content = """
        model:
          num_residual_blocks: 2
          num_filters: 64
        mcts:
          c_puct: 1.0
          simulations_per_move: 10
        training:
          learning_rate: 0.001
          batch_size: 32
          num_iterations: 2
          num_self_play_games_per_iteration: 5
          replay_buffer_capacity: 100
          num_training_steps: 50
          use_mixed_precision: True
          l2_regularization_coeff: 0.0001
        checkpointing:
          checkpoint_dir: "checkpoints"
          checkpoint_frequency: 1
          load_checkpoint: False
          load_checkpoint_path: ""
        logging:
          tensorboard_log_dir: "runs/alpha_chess_experiment"
        testing: # New section for testing specific configs
          use_torch_compile: True # New parameter for torch.compile in testing
        """
        with open(config_path, "w") as f:
            f.write(dummy_config_content)

    tester = Tester()

    # Example usage:
    # Load a trained model (replace with actual path to a .pth file)
    # tester._load_model("checkpoints/model_iter_1.pth")

    # Run self-play test games
    # tester.run_self_play_test(num_games=5)

    # Run human vs agent CLI game
    tester.run_human_play_test()
