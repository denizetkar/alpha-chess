from typing import TypedDict, Optional


class ModelConfig(TypedDict):
    num_residual_blocks: int
    num_filters: int
    use_torch_compile: bool


class LRSchedulerConfig(TypedDict):
    use_scheduler: bool
    type: str
    t_max: int
    eta_min: float
    gamma: float


class TrainingConfig(TypedDict):
    num_iterations: int
    num_games_per_iteration: int
    num_epochs: int
    batch_size: int
    num_training_steps: int
    learning_rate: float
    l2_regularization: float
    momentum: float
    clip_norm: int
    value_loss_weight: float
    policy_loss_weight: float
    temperature: float
    cpu_threads: int
    replay_buffer_capacity: int
    lr_scheduler: LRSchedulerConfig
    use_mixed_precision: bool
    seed: Optional[int]


class MCTSConfig(TypedDict):
    num_simulations: int
    c_puct: float
    temp_threshold: int
    max_depth: int
    simulations_per_move: int
    dirichlet_alpha: float
    dirichlet_epsilon: float


class CheckpointingConfig(TypedDict):
    checkpoint_dir: str
    save_interval: int
    log_interval: int
    checkpoint_path: str
    load_checkpoint: bool
    load_checkpoint_path: Optional[str]


class LoggingConfig(TypedDict):
    tensorboard_log_dir: str


class TrainFullConfig(TypedDict):
    model: ModelConfig
    training: TrainingConfig
    mcts: MCTSConfig
    checkpointing: CheckpointingConfig
    logging: LoggingConfig
    device: str


class TestingConfig(TypedDict):
    # This config is currently empty as use_torch_compile was moved to ModelConfig
    # If other testing-specific parameters are added, they would go here.
    pass


class TestFullConfig(TypedDict):
    model: ModelConfig
    mcts: MCTSConfig
    checkpointing: CheckpointingConfig
    testing: TestingConfig
