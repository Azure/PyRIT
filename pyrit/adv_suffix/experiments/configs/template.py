import os
from ml_collections import config_dict
from pyrit.common import default_values



def get_config():
    # Load HuggingFace token from .env
    default_values.load_default_env()
    token = os.environ.get("HF_TOKEN")

    config = config_dict.ConfigDict()

    # Experiment type
    config.transfer = False

    # General parameters
    config.target_weight = 1.0
    config.control_weight = 0.0
    config.progressive_goals = False
    config.progressive_models = False
    config.anneal = False
    config.incr_control = False
    config.stop_on_success = False
    config.verbose = True
    config.allow_non_ascii = False
    config.num_train_models = 1

    # Results
    config.result_prefix = "results/individual_vicuna7b"

    # tokenizers
    config.tokenizer_paths = ["/data/vicuna/vicuna-7b-v1.3"]
    config.tokenizer_kwargs = [{"use_fast": False}]

    config.model_paths = ["/data/vicuna/vicuna-7b-v1.3"]
    config.token = token
    config.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False}]
    config.conversation_templates = ["vicuna"]
    config.devices = ["cuda:0"]

    # data
    config.train_data = ""
    config.test_data = ""
    config.n_train_data = 50
    config.n_test_data = 0
    config.data_offset = 0

    # attack-related parameters
    config.attack = "gcg"
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.n_steps = 500
    config.test_steps = 50
    config.batch_size = 512
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    config.filter_cand = True

    config.gbda_deterministic = True

    return config
