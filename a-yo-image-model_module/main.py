from data_utils import *
from model_utils import *
from function_utils import *

"""
Args json
- data_path:
- model_name:
"""


def main():
    ################# Load Data #################
    data = data_load(data_path)
    dataset = dataset_packing(data)
    
    ################# Load Model #################
    model = load_model_(model_name, summary=True)
    generated_frames = predict(model, dataset)
    
    
if __name__ == "__main__":
    main()
    
    
    