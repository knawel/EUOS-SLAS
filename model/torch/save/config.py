from datetime import datetime

config_data = {
    'dataset_filepath': "data/data_seq_locations.xz"
}

# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v1' + tag,
    'output_dir': 'save',
    'device': 'cuda',
    'num_epochs': 70,
    'batch_size': 256,
    'log_step': 1024,
    'learning_rate': 1e-5,
    'hidden_size': 128,
    'layers': 1
}