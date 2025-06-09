import argparse
import torch
import chess
import yaml
import os  # Import os
import random  # Import random
import numpy as np  # Import numpy
from tqdm import tqdm  # Import tqdm
from src.chess_env import ChessEnv
from src.nn_model import AlphaChessNet
from src.mcts import MCTSNode, MCTS
from src.move_encoder import MoveEncoderDecoder, MOVE_ENCODING_SIZE
from src.config_types import TestFullConfig


def load_config(config_path: str) -> TestFullConfig:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration.
    """
    try:
        if not os.path.exists(config_path):
            return {
                "model": {"num_residual_blocks": 2, "num_filters": 128, "use_torch_compile": False},
                "mcts": {
                    "num_simulations": 10,
                    "c_puct": 1.0,
                    "temp_threshold": 1,
                    "max_depth": 2,
                    "simulations_per_move": 10,
                    "dirichlet_alpha": 0.3,
                    "dirichlet_epsilon": 0.25,
                },
                "checkpointing": {
                    "checkpoint_dir": "checkpoints",
                    "save_interval": 1,
                    "log_interval": 1,
                    "checkpoint_path": "checkpoints/model.pth",
                    "load_checkpoint": False,
                    "load_checkpoint_path": None,
                },
                "testing": {},
            }

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:  # Handle empty config file
                return load_config("")  # Recursively call with empty path to get defaults
            return config
    except yaml.YAMLError:
        return load_config("")  # Recursively call with empty path to get defaults
    except FileNotFoundError:
        exit(1)


class Tester:
    """
    A class for testing the AlphaChess agent in self-play or human-play modes.
    It handles model loading, game simulation, and interaction with human players.
    """

    def __init__(self, config: TestFullConfig):
        """
        Initializes the Tester with a given configuration.

        Args:
            config (dict): A dictionary containing the testing configuration,
                           including model, MCTS, and checkpointing settings.
        """
        self.config: TestFullConfig = config
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.chess_env: ChessEnv = ChessEnv()
        self.move_encoder: MoveEncoderDecoder = MoveEncoderDecoder()
        self.model: AlphaChessNet = AlphaChessNet(
            num_residual_blocks=self.config["model"]["num_residual_blocks"],
            num_filters=self.config["model"]["num_filters"],
        ).to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Compile the model if PyTorch 2.0+ is available and configured
        if self.config["model"]["use_torch_compile"] and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore

        self.checkpoint_dir = self.config["checkpointing"]["checkpoint_dir"]

    def _load_model(self, checkpoint_path: str) -> None:
        """
        Loads the model state from a given checkpoint path.

        Args:
            checkpoint_path (str): The file path to the model checkpoint.
        """
        checkpoint: dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore

    def _play_game_with_agent(self, agent_color: chess.Color) -> tuple[str, list[chess.Move]]:
        """
        Simulates a single game where the agent plays against a human or another agent.

        Args:
            agent_color (chess.Color): The color (chess.WHITE or chess.BLACK) the agent plays as.

        Returns:
            tuple[str, list[chess.Move]]: A tuple containing the game result string (e.g., '1-0', '0-1', '1/2-1/2')
                                          and a list of moves made during the game.
        """
        self.chess_env.reset()
        current_board: chess.Board = self.chess_env.board.copy()
        mcts: MCTS = MCTS(
            self.model,
            self.chess_env,
            self.move_encoder,
            self.device,  # Pass device
            c_puct=self.config["mcts"]["c_puct"],
        )

        game_moves: list[chess.Move] = []
        while not current_board.is_game_over():
            if current_board.turn == agent_color:
                # Agent's turn
                root_node: MCTSNode = MCTSNode(current_board)
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

    def run_self_play_test(self, num_games: int) -> None:
        """
        Runs a specified number of self-play test games.

        Args:
            num_games (int): The number of self-play games to run.
        """
        results: dict[str, int] = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        for i in tqdm(range(num_games), desc="Self-Play Test"):
            self.chess_env.reset()
            current_board = self.chess_env.board.copy()
            root_node = MCTSNode(current_board)  # Initialize root_node once per game
            mcts = MCTS(
                self.model,
                self.chess_env,
                self.move_encoder,
                self.device,  # Pass device
                c_puct=self.config["mcts"]["c_puct"],
            )

            while not current_board.is_game_over():
                legal_moves_list = list(current_board.legal_moves)

                mcts.run_simulations(root_node, self.config["mcts"]["simulations_per_move"])

                # print(f"MCTS Root Node Q: {root_node.Q}")

                chosen_move = None
                if root_node.N > 0:
                    policy_target = np.zeros(MOVE_ENCODING_SIZE)
                    for move, child_node in root_node.children.items():
                        move_idx = self.move_encoder.encode(current_board, move)
                        if move_idx < MOVE_ENCODING_SIZE:
                            policy_target[move_idx] = child_node.N

                    # print(f"Policy Target (before norm): {policy_target}")

                    sum_visits = np.sum(policy_target)
                    if sum_visits > 0:
                        policy_target = policy_target / sum_visits
                    else:
                        if legal_moves_list:
                            for move in legal_moves_list:
                                move_idx = self.move_encoder.encode(current_board, move)
                                if move_idx < MOVE_ENCODING_SIZE:
                                    policy_target[move_idx] = 1.0 / len(legal_moves_list)
                        else:
                            chosen_move = None  # No legal moves, game should be over

                    if chosen_move is None:  # Only proceed if a move can potentially be chosen
                        if self.chess_env.board.fullmove_number <= self.config["mcts"]["temp_threshold"]:
                            # Add a small epsilon to probabilities to avoid issues with np.random.choice on zero sums
                            temp_policy = np.power(policy_target, 1.0 / 1.0)
                            temp_policy_sum = np.sum(temp_policy)
                            if temp_policy_sum > 0:
                                temp_policy /= temp_policy_sum
                                chosen_move_idx = int(np.random.choice(len(temp_policy), p=temp_policy))
                            else:
                                chosen_move_idx = -1  # Indicate no valid choice
                        else:
                            chosen_move_idx = int(np.argmax(policy_target))

                        if chosen_move_idx != -1:
                            chosen_move = self.move_encoder.decode(chosen_move_idx)
                        else:
                            chosen_move = None  # No valid move could be chosen from policy

                    if chosen_move is None or chosen_move not in current_board.legal_moves:
                        if legal_moves_list:
                            chosen_move = random.choice(legal_moves_list)
                        else:
                            chosen_move = None
                else:
                    if legal_moves_list:
                        chosen_move = random.choice(legal_moves_list)
                    else:
                        chosen_move = None

                if chosen_move is None:
                    break

                current_board.push(chosen_move)
                self.chess_env.board = current_board.copy()
                # Update root_node to the child corresponding to the chosen_move
                # If the child doesn't exist (e.g., due to random fallback), create a new node
                root_node = root_node.children.get(chosen_move, MCTSNode(current_board))
                root_node.parent = None  # Detach from old tree

            game_result = current_board.result()
            results[game_result] += 1

        print("\nSelf-Play Test Results:")
        for res, count in results.items():
            print(f"{res}: {count} games")
        print(f"Total games: {num_games}")

    def run_human_play_test(self, agent_color: chess.Color) -> None:
        """
        Runs a human vs. AlphaChess game in a command-line interface.

        Args:
            agent_color (chess.Color): The color (chess.WHITE or chess.BLACK) the agent plays as.
        """
        print("\n--- Starting Human vs AlphaChess (CLI) ---")
        print("Enter moves in UCI format (e.g., 'e2e4'). Type 'quit' to exit.")
        print(f"AlphaChess agent plays as: {'White' if agent_color == chess.WHITE else 'Black'}")

        self.chess_env.reset()
        current_board = self.chess_env.board.copy()
        mcts = MCTS(
            self.model,
            self.chess_env,
            self.move_encoder,
            self.device,  # Pass device
            c_puct=self.config["mcts"]["c_puct"],
        )

        while not current_board.is_game_over():
            print("\n" + str(current_board))
            print(f"Turn: {'White' if current_board.turn == chess.WHITE else 'Black'}")

            if current_board.turn == agent_color:
                # Agent's turn
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
            else:  # Human's turn
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

            self.chess_env.board = current_board.copy()  # Update env board

        print("\nGame Over!")
        print(f"Result: {current_board.result()}")


def _apply_cli_overrides(config: TestFullConfig, args: argparse.Namespace) -> TestFullConfig:
    """
    Applies command-line arguments to override configuration values.

    Args:
        config (dict): The base configuration dictionary.
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        dict: The updated configuration dictionary with CLI overrides applied.
    """
    if args.simulations_per_move is not None:
        config["mcts"]["simulations_per_move"] = args.simulations_per_move
    if args.c_puct is not None:
        config["mcts"]["c_puct"] = args.c_puct
    if args.use_torch_compile is not None:
        config["model"]["use_torch_compile"] = args.use_torch_compile

    # Ensure checkpoint_path is set in config for Tester to use
    config["checkpointing"]["load_checkpoint_path"] = args.checkpoint_path

    return config


def main() -> None:
    """
    Main function for the AlphaChess testing script.
    Handles argument parsing, configuration loading, and execution of
    self-play or human-play test modes.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="AlphaChess Testing Script")

    # Mode selection
    mode_group: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--self-play",
        action="store_true",
        help="Run self-play test games.",
    )
    mode_group.add_argument(
        "--human-play",
        action="store_true",
        help="Run human vs. AlphaChess game.",
    )

    # General configuration
    parser.add_argument(
        "--config_path",
        type=str,
        default="test_config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--use_torch_compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile for model optimization (default: False).",
    )

    # Self-play specific arguments
    self_play_group = parser.add_argument_group("Self-Play Options")
    self_play_group.add_argument(
        "--num_games",
        type=int,
        default=1,  # Changed to 1 for detailed debugging
        help="Number of self-play games to run.",
    )

    # Human-play specific arguments
    human_play_group = parser.add_argument_group("Human-Play Options")
    human_play_group.add_argument(
        "--agent_color",
        type=str,
        choices=["white", "black"],
        default="black",
        help="Color of the AlphaChess agent in human-play mode (white or black).",
    )

    # MCTS overrides
    mcts_group = parser.add_argument_group("MCTS Overrides")
    mcts_group.add_argument(
        "--simulations_per_move",
        type=int,
        help="Override the number of MCTS simulations per move.",
    )
    mcts_group.add_argument(
        "--c_puct",
        type=float,
        help="Override the MCTS C_PUCT value.",
    )

    args = parser.parse_args()

    config = load_config(args.config_path)
    config = _apply_cli_overrides(config, args)

    tester = Tester(config)  # Pass the config directly

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        tester._load_model(args.checkpoint_path)
    else:
        print(f"Warning: Checkpoint path '{args.checkpoint_path}' not found. Using randomly initialized model.")

    try:
        if args.self_play:
            tester.run_self_play_test(num_games=args.num_games)
        elif args.human_play:
            tester.run_human_play_test(agent_color=chess.WHITE if args.agent_color == "white" else chess.BLACK)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
