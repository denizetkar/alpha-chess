import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import yaml
import os
from collections import deque
import random
import numpy as np
import chess
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.chess_env import ChessEnv
from src.nn_model import AlphaChessNet
from src.mcts import MCTSNode, MCTS
from src.move_encoder import MoveEncoderDecoder, MOVE_ENCODING_SIZE


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.chess_env = ChessEnv()
        self.move_encoder = MoveEncoderDecoder()
        self.model: torch.nn.Module = AlphaChessNet(  # Explicitly type as torch.nn.Module
            num_residual_blocks=self.config["model"]["num_residual_blocks"],
            num_filters=self.config["model"]["num_filters"],
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["l2_regularization_coeff"],
        )  # type: ignore
        self.scaler = GradScaler()

        # Compile the model if PyTorch 2.0+ is available and configured
        if self.config["training"]["use_torch_compile"] and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)  # type: ignore

        # Learning rate scheduler
        if self.config["training"]["lr_scheduler"]["use_scheduler"]:
            if self.config["training"]["lr_scheduler"]["type"] == "cosine_annealing":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config["training"]["lr_scheduler"]["t_max"],
                    eta_min=self.config["training"]["lr_scheduler"]["eta_min"],
                )
            elif self.config["training"]["lr_scheduler"]["type"] == "exponential":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=self.config["training"]["lr_scheduler"]["gamma"],
                )
            else:
                raise ValueError(f"Unknown LR scheduler type: {self.config['training']['lr_scheduler']['type']}")
        else:
            self.scheduler = None

        self.replay_buffer = deque(maxlen=self.config["training"]["replay_buffer_capacity"])
        self.writer = SummaryWriter(log_dir=self.config["logging"]["tensorboard_log_dir"])

        self.checkpoint_dir = self.config["checkpointing"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, iteration):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_iter_{iteration}.pth")
        try:
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": self.model.state_dict(),  # type: ignore
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "config": self.config,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at {checkpoint_path}")
        except IOError as e:
            print(f"Error saving checkpoint to {checkpoint_path}: {e}")

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint.get("iteration", 0)
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
            return 0
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}. Starting from scratch.")
            return 0

    def _apply_dirichlet_noise(self, root_node, current_board, dirichlet_alpha, dirichlet_epsilon):
        """Applies Dirichlet noise to the prior probabilities of legal moves."""
        prior_probs = np.zeros(MOVE_ENCODING_SIZE)
        legal_moves_indices = []
        for move in root_node.legal_moves:
            move_idx = self.move_encoder.encode(current_board, move)
            if move_idx < MOVE_ENCODING_SIZE:
                prior_probs[move_idx] = root_node.P.get(move, 0.0) if root_node.P is not None else 0.0
                legal_moves_indices.append(move_idx)

        if legal_moves_indices:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves_indices))
            for i, idx in enumerate(legal_moves_indices):
                prior_probs[idx] = (1 - dirichlet_epsilon) * prior_probs[idx] + dirichlet_epsilon * noise[i]

            sum_noisy_probs = np.sum(prior_probs[legal_moves_indices])
            if sum_noisy_probs > 0:
                prior_probs[legal_moves_indices] /= sum_noisy_probs
            else:
                prior_probs[legal_moves_indices] = 1.0 / len(legal_moves_indices)
        return prior_probs

    def _get_policy_target_and_chosen_move(self, root_node, current_board, prior_probs):
        """Calculates policy target and selects a move based on MCTS visit counts and priors."""
        policy_target = np.zeros(MOVE_ENCODING_SIZE)
        for move, child_node in root_node.children.items():
            move_idx = self.move_encoder.encode(current_board, move)
            if move_idx < MOVE_ENCODING_SIZE:
                policy_target[move_idx] = child_node.N

        sum_visits = np.sum(policy_target)
        if sum_visits > 0:
            policy_target = policy_target / sum_visits
        else:
            policy_target = prior_probs

        chosen_move = None
        policy_target_sum = np.sum(policy_target)
        if policy_target_sum == 0:
            legal_moves_list = list(current_board.legal_moves)
            if legal_moves_list:
                policy_target = np.zeros(MOVE_ENCODING_SIZE)
                for move in legal_moves_list:
                    move_idx = self.move_encoder.encode(current_board, move)
                    policy_target[move_idx] = 1.0 / len(legal_moves_list)
            else:
                return policy_target, None  # No legal moves, game over

        chosen_move_idx = np.random.choice(len(policy_target), p=policy_target)
        chosen_move = self.move_encoder.decode(chosen_move_idx)

        if chosen_move is None or chosen_move not in current_board.legal_moves:
            legal_moves_list = list(current_board.legal_moves)
            if legal_moves_list:
                chosen_move = random.choice(legal_moves_list)
            else:
                return policy_target, None  # No legal moves, game over
        return policy_target, chosen_move

    def _self_play_game(self):
        self.chess_env.reset()
        game_states = []
        current_board = self.chess_env.board.copy()
        root_node = MCTSNode(current_board)
        mcts = MCTS(
            self.model,
            self.chess_env,
            self.move_encoder,
            c_puct=self.config["mcts"]["c_puct"],
            max_depth=self.config["mcts"]["max_depth"],  # Pass max_depth from config
        )  # type: ignore

        while not current_board.is_game_over():
            mcts.run_simulations(root_node, self.config["mcts"]["simulations_per_move"])

            chosen_move = None
            policy_target = (
                np.ones(MOVE_ENCODING_SIZE) / MOVE_ENCODING_SIZE
            )  # Initialize policy_target to uniform distribution

            if root_node.N > 0:
                # Add Dirichlet noise to the root node's prior probabilities for exploration
                # This is applied to the policy probabilities before selecting a move.
                dirichlet_alpha = self.config["mcts"]["dirichlet_alpha"]
                dirichlet_epsilon = self.config["mcts"]["dirichlet_epsilon"]

                prior_probs = self._apply_dirichlet_noise(root_node, current_board, dirichlet_alpha, dirichlet_epsilon)
                policy_target, chosen_move = self._get_policy_target_and_chosen_move(
                    root_node, current_board, prior_probs
                )
                if chosen_move is None:
                    break  # No legal moves or problem during selection, end game
            else:
                # If MCTS couldn't find any moves (e.g., root_node.N is 0),
                # this implies a problem or terminal state.
                # Fallback: choose a random legal move.
                legal_moves_list = list(current_board.legal_moves)
                if legal_moves_list:
                    chosen_move = random.choice(legal_moves_list)
                else:
                    break  # No legal moves, game over

            # Ensure chosen_move is not None before pushing
            if chosen_move is None:
                break  # Should not happen with fallbacks, but as a safeguard

            game_states.append((self.chess_env.get_state_planes(), policy_target, current_board.turn))

            current_board.push(chosen_move)
            self.chess_env.board = current_board.copy()

            if chosen_move in root_node.children:
                root_node = root_node.children[chosen_move]
                root_node.parent = None
            else:
                root_node = MCTSNode(current_board)

        game_result = self.chess_env.result()
        outcome_value = 0.0
        if game_result == "1-0":
            outcome_value = 1.0
        elif game_result == "0-1":
            outcome_value = -1.0

        final_game_data = []
        for state_planes, policy_target, turn_at_state in game_states:
            value_for_state = outcome_value
            if turn_at_state == chess.BLACK:
                value_for_state = -outcome_value
            final_game_data.append((state_planes, policy_target, value_for_state))

        return final_game_data

    def train(self):
        num_iterations = self.config["training"]["num_iterations"]
        games_per_iteration = self.config["training"]["num_self_play_games_per_iteration"]
        batch_size = self.config["training"]["batch_size"]
        num_training_steps = self.config["training"]["num_training_steps"]

        start_iteration = 0
        if self.config["checkpointing"]["load_checkpoint"]:
            start_iteration = self._load_checkpoint(self.config["checkpointing"]["load_checkpoint_path"])

        try:
            for iteration in range(start_iteration, num_iterations):
                print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

                # Self-play phase
                print("Generating self-play games...")
                for _ in tqdm(range(games_per_iteration), desc="Self-play"):
                    try:
                        game_data = self._self_play_game()
                        self.replay_buffer.extend(game_data)
                    except Exception as e:
                        print(f"Error during self-play game generation: {e}. Skipping this game.")
                        continue

                # Training phase
                print("Training neural network...")
                if len(self.replay_buffer) < batch_size:
                    print("Replay buffer too small for training. Skipping training this iteration.")
                    continue

                for step in tqdm(range(num_training_steps), desc="Training"):
                    self._perform_training_step(batch_size, iteration, num_training_steps, step)

                # Save checkpoint
                if (iteration + 1) % self.config["checkpointing"]["checkpoint_frequency"] == 0:
                    self._save_checkpoint(iteration + 1)

        except Exception as e:
            print(f"An unexpected error occurred during training: {e}")
        finally:
            self.writer.close()
            print("Training complete (or terminated due to error).")

    def _perform_training_step(self, batch_size, iteration, num_training_steps, step):
        """Performs a single training step."""
        try:
            batch = random.sample(self.replay_buffer, batch_size)
            states, policies, values = zip(*batch)

            states_tensor = torch.from_numpy(np.array(states)).float().to(self.device)
            policies_tensor = torch.from_numpy(np.array(policies)).float().to(self.device)
            values_tensor = torch.tensor(values).float().unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.config["training"]["use_mixed_precision"]):
                pred_policy_logits, pred_value = self.model(states_tensor)

                policy_loss = -torch.sum(policies_tensor * pred_policy_logits, dim=1).mean()
                value_loss = F.mse_loss(pred_value, values_tensor)

                # L2 regularization is now handled by weight_decay in the optimizer
                total_loss = policy_loss + value_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step the learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            self.writer.add_scalar("Loss/Policy", policy_loss.item(), iteration * num_training_steps + step)
            self.writer.add_scalar("Loss/Value", value_loss.item(), iteration * num_training_steps + step)
            self.writer.add_scalar("Loss/Total", total_loss.item(), iteration * num_training_steps + step)
            self.writer.add_scalar(
                "LearningRate", self.optimizer.param_groups[0]["lr"], iteration * num_training_steps + step
            )
        except Exception as e:
            print(f"Error during training step {step}: {e}. Skipping this step.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train AlphaChess model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override learning rate from config.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Override number of training iterations from config.",
    )
    parser.add_argument(
        "--simulations_per_move",
        type=int,
        help="Override MCTS simulations per move from config.",
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        help="Path to a checkpoint to load. Sets load_checkpoint to True.",
    )
    parser.add_argument(
        "--use_mixed_precision",
        type=bool,
        help="Override use_mixed_precision from config (True/False).",
    )
    parser.add_argument(
        "--use_torch_compile",
        type=bool,
        help="Override use_torch_compile from config (True/False).",
    )
    # Add more arguments as needed for common overrides

    args = parser.parse_args()

    # Load config
    config_path = args.config
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. Please create one or specify a valid path."
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading configuration file {config_path}: {e}")
        return  # Exit if config cannot be loaded

    # Apply command-line overrides
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_iterations is not None:
        config["training"]["num_iterations"] = args.num_iterations
    if args.simulations_per_move is not None:
        config["mcts"]["simulations_per_move"] = args.simulations_per_move
    if args.load_checkpoint_path is not None:
        config["checkpointing"]["load_checkpoint"] = True
        config["checkpointing"]["load_checkpoint_path"] = args.load_checkpoint_path
    if args.use_mixed_precision is not None:
        config["training"]["use_mixed_precision"] = args.use_mixed_precision
    if args.use_torch_compile is not None:
        config["training"]["use_torch_compile"] = args.use_torch_compile

    # Set random seeds for reproducibility
    if "seed" in config["training"] and config["training"]["seed"] is not None:
        seed = config["training"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")

    # Pass the modified config to the Trainer
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
