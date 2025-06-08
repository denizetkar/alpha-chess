from typing import TypedDict, Optional


class ModelConfig(TypedDict):
    num_residual_blocks: int
    num_filters: int
    use_torch_compile: bool  # Added for train_config and test_config


class LRSchedulerConfig(TypedDict):
    use_scheduler: bool
    type: str
    t_max: int
    eta_min: float
    gamma: float


class TrainingConfig(TypedDict):
    num_iterations: int
    num_games_per_iteration: int  # Renamed from num_self_play_games_per_iteration
    num_epochs: int  # Added
    batch_size: int
    num_training_steps: int
    learning_rate: float
    l2_regularization: float  # Renamed from l2_regularization_coeff
    momentum: float  # Added
    clip_norm: int  # Added
    value_loss_weight: float  # Added
    policy_loss_weight: float  # Added
    temperature: float  # Added
    cpu_threads: int  # Added
    replay_buffer_capacity: int
    lr_scheduler: LRSchedulerConfig
    use_mixed_precision: bool  # Added
    use_torch_compile: bool
    seed: Optional[int]


class MCTSConfig(TypedDict):
    num_simulations: int  # Added
    c_puct: float
    temp_threshold: int  # Added
    max_depth: int
    simulations_per_move: int
    dirichlet_alpha: float  # Added for train_config
    dirichlet_epsilon: float  # Added for train_config


class CheckpointingConfig(TypedDict):
    checkpoint_dir: str
    save_interval: int  # Added
    log_interval: int  # Added
    checkpoint_path: str  # Added
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
    # These fields are present in test_config.yaml but are part of other sections in train_config.yaml
    # For test_config, they are directly under the root or 'testing' section.
    # I will define a structure that allows for partial configs to be loaded.
    # For the 'test.py' script, it expects 'model', 'mcts', 'checkpointing' and 'testing' at the top level.
    # So, I will define a TestFullConfig that mirrors this structure.
    use_torch_compile: bool


class TestFullConfig(TypedDict):
    model: ModelConfig
    mcts: MCTSConfig
    checkpointing: CheckpointingConfig
    testing: TestingConfig
