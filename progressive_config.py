"""
Configuration for progressive training
"""


class ProgressiveTrainingConfig:
    """Configuration parameters for progressive training with cross-validation"""
    
    # Cross-validation settings
    NUM_FOLDS = 5
    RANDOM_SEED = 42
    VALIDATION_SPLIT = 0.2  # For single model training (non-CV)
    
    # Model architecture
    ENCODER_NAME = "efficientnet-b5"
    ENCODER_PRETRAIN_WEIGHTS = "imagenet"
    INPUT_CHANNELS = 3
    OUTPUT_CLASSES = 1  # Binary segmentation
    
    # Progressive training phases
    # Each phase: image_size, num_epochs, learning_rate, batch_size
    TRAINING_PHASES = [
        {
            'image_size': 256,
            'num_epochs': 50,
            'initial_learning_rate': 0.001,
            'batch_size': 16,
            'phase_name': 'Phase 1: Small Images (Fast Learning)'
        },
        {
            'image_size': 512,
            'num_epochs': 50,
            'initial_learning_rate': 0.0001,
            'batch_size': 8,
            'phase_name': 'Phase 2: Medium Images (Refinement)'
        },
        {
            'image_size': 768,
            'num_epochs': 50,
            'initial_learning_rate': 0.00001,
            'batch_size': 4,
            'phase_name': 'Phase 3: Large Images (Fine-tuning)'
        },
    ]
    
    # DataLoader settings
    NUM_WORKERS = 0  # Set to 0 for macOS compatibility
    PIN_MEMORY = False
    
    # Output directories
    RESULTS_OUTPUT_DIR = "progressive_results"
    MODELS_OUTPUT_DIR = "progressive_models"
    
    # Scheduler settings
    COSINE_ANNEALING_T_MAX = None  # Will be set to num_epochs per phase
    
    # Early stopping (optional)
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    @classmethod
    def get_total_epochs(cls):
        """Calculate total number of epochs across all phases"""
        return sum(phase['num_epochs'] for phase in cls.TRAINING_PHASES)
    
    @classmethod
    def get_phase_info(cls, phase_index):
        """Get information about a specific phase"""
        if 0 <= phase_index < len(cls.TRAINING_PHASES):
            return cls.TRAINING_PHASES[phase_index]
        return None
