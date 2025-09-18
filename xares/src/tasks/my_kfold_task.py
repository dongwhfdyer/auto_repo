from xares.task import TaskConfig


def my_kfold_config(encoder) -> TaskConfig:
    config = TaskConfig(
        # Basic task info
        name="my_kfold_task",
        formal_name="My K-Fold Dataset",
        encoder=encoder,
        
        # IMPORTANT: Set this to True since you have local data
        private=True,  # Prevents xares from trying to download from Zenodo
        
        # IMPORTANT: Set this to the parent directory of your task folder
        # If you generated tars in /path/to/env_root/my_task/
        # Then set env_root to /path/to/env_root
        env_root="/path/to/env_root",  # UPDATE THIS PATH!
        
        # K-FOLD CONFIGURATION - This is the key part!
        k_fold_splits=list(range(1, 6)),  # 5 folds: 1, 2, 3, 4, 5
        # This must match the fold numbers in your generated tar files
        
        # Model configuration
        output_dim=2,  # Binary classification: normal=0, abnormal=1
        metric="accuracy",  # or "f1", "precision", "recall", etc.
        
        # Training hyperparameters
        epochs=10,
        batch_size_train=32,
        batch_size_valid=32,
        learning_rate=1e-3,
        
        # Label processing - extracts the label from your data
        label_processor=lambda x: x["label"],
        
        # Optional: Enable/disable KNN evaluation
        do_knn=True,
    )
    
    # Optional: Override tar naming patterns if needed
    # The default patterns should work with your generated files:
    # config.audio_tar_name_of_split = {fold: f"wds-audio-fold-{fold}-*.tar" for fold in config.k_fold_splits}
    # config.encoded_tar_name_of_split = {fold: f"wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits}
    
    return config
