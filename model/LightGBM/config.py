from datetime import datetime

config_data = {
    'dataset_filepath': "../../data/preprocessed"
}

# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v1' + tag
}
