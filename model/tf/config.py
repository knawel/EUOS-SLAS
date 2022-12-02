from datetime import datetime


# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v1' + tag,
    'output_dir': 'save',
    'device': 'cuda',
    'num_epochs': 200,
    'batch_size': 2048,
    'log_step': 1024,
    'learning_rate': 1e-5
}
