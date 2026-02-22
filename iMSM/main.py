import iMSM.data_classes.config_data_class as config_data_class
from iMSM.data_classes.input_data_classes import iMSMInput
from iMSM.data_classes.config_data_class import iMSMConfig
from iMSM._1_categorize import default_categorize
from iMSM._2_embed import default_embed
from iMSM._3_cluster import default_cluster
from iMSM._4_msm import default_msm
import pickle
import os

STAGE_STRINGS = {
    1: "1_categorization",
    2: "2_embedding",
    3: "3_clustering",
    4: "4_msm"
}

STAGE_DEFAULT_FUNCTIONS = {
    1: default_categorize,
    2: default_embed,
    3: default_cluster,
    4: default_msm
}

def run(input: iMSMInput, config: iMSMConfig):
    # Create base checkpoints directory if it doesn't exist
    os.makedirs(config.checkpoints_path, exist_ok=True)
        
    
    # Determine starting point
    if config.start_stage == 1:
        previous_stage_output = input
    else:
        previous_stage_output = load_data_class(config.checkpoints_path, config.start_stage - 1)
        
    # Run through stages
    for stage in range(config.start_stage, config.end_stage):
        print(f"--- Running stage {STAGE_STRINGS[stage]} ---")
        
        stage_function = config.__dict__.get(f"custom_{STAGE_STRINGS[stage]}", None) or STAGE_DEFAULT_FUNCTIONS[stage]
        current_stage_output = stage_function(previous_stage_output, config)
        save_data_class(current_stage_output, config.checkpoints_path, stage)
        previous_stage_output = current_stage_output

def save_data_class(data_class_instance, base_filename: str, stage: int):
    with open(f"{base_filename}/{STAGE_STRINGS[stage]}.pickle", "wb") as f:
        pickle.dump(data_class_instance, f)
        
def load_data_class(base_filename: str, stage: int):
    with open(f"{base_filename}/{STAGE_STRINGS[stage]}.pickle", "rb") as f:
        return pickle.load(f)
    