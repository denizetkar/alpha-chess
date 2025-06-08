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
from src.move_encoder import MoveEncoderDecoder


class Trainer:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.chess_env = ChessEnv()
        self.move_encoder = MoveEncoderDecoder()
        self.model = AlphaChessNet(
            num_residual_blocks=self.config["model"]["num_residual_blocks"],
            num_filters=self.config["model"]["num_filters"],
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["training"]["learning_rate"])
        self.scaler = GradScaler()

        self.replay_buffer = deque(maxlen=self.config["training"]["replay_buffer_capacity"])
        self.writer = SummaryWriter(log_dir=self.config["logging"]["tensorboard_log_dir"])

        self.checkpoint_dir = self.config["checkpointing"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _save_checkpoint(self, iteration):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_iter_{iteration}.pth")
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get("iteration", 0)

    def _self_play_game(self):
        self.chess_env.reset()
        game_states = []
        current_board = self.chess_env.board.copy()
        root_node = MCTSNode(current_board)
        mcts = MCTS(self.model, self.chess_env, self.move_encoder, c_puct=self.config["mcts"]["c_puct"])

        while not current_board.is_game_over():
            mcts.run_simulations(root_node, self.config["mcts"]["simulations_per_move"])

            chosen_move = None
            policy_target = np.ones(4672) / 4672.0  # Initialize policy_target to uniform distribution

            if root_node.N > 0:
                # Collect visit counts for legal moves
                temp_policy_target = np.zeros(4672)
                for move, child_node in root_node.children.items():
                    # Pass the current_board to the encoder
                    move_idx = self.move_encoder.encode(current_board, move)
                    if move_idx < 4672:
                        temp_policy_target[move_idx] = child_node.N

                sum_visits = np.sum(temp_policy_target)
                if sum_visits > 0:
                    policy_target = temp_policy_target / sum_visits
                # else: policy_target remains uniform as initialized

                # Select move based on policy target (stochastic for self-play)
                chosen_move_idx = np.random.choice(len(policy_target), p=policy_target)
                chosen_move = self.move_encoder.decode(chosen_move_idx)

                # Fallback if chosen_move is None (due to encoder placeholder) or illegal
                if chosen_move is None or chosen_move not in current_board.legal_moves:
                    legal_moves_list = list(current_board.legal_moves)
                    if legal_moves_list:
                        chosen_move = random.choice(legal_moves_list)
                    else:
                        # No legal moves, game should be over (stalemate/checkmate)
                        break
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

        for iteration in range(start_iteration, num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

            # Self-play phase
            print("Generating self-play games...")
            for _ in tqdm(range(games_per_iteration), desc="Self-play"):
                game_data = self._self_play_game()
                self.replay_buffer.extend(game_data)

            # Training phase
            print("Training neural network...")
            if len(self.replay_buffer) < batch_size:
                print("Replay buffer too small for training. Skipping training this iteration.")
                continue

            for step in tqdm(range(num_training_steps), desc="Training"):
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

                    # L2 regularization (already handled by optimizer weight_decay if set)
                    # total_loss = policy_loss + value_loss + \
                    # self.config["training"]["l2_regularization_coeff"] * \
                    # sum(p.pow(2).sum() for p in self.model.parameters())
                    total_loss = policy_loss + value_loss

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.writer.add_scalar("Loss/Policy", policy_loss.item(), iteration * num_training_steps + step)
                self.writer.add_scalar("Loss/Value", value_loss.item(), iteration * num_training_steps + step)
                self.writer.add_scalar("Loss/Total", total_loss.item(), iteration * num_training_steps + step)
                self.writer.add_scalar(
                    "LearningRate", self.optimizer.param_groups[0]["lr"], iteration * num_training_steps + step
                )

            # Save checkpoint
            if (iteration + 1) % self.config["checkpointing"]["checkpoint_frequency"] == 0:
                self._save_checkpoint(iteration + 1)

        self.writer.close()
        print("Training complete.")


if __name__ == "__main__":
    # Create a dummy config.yaml for testing
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
    """
    with open("config.yaml", "w") as f:
        f.write(dummy_config_content)

    trainer = Trainer()
    trainer.train()
