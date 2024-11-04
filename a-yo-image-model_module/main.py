from data_utils import *
from model_utils import *
from function_utils import *
from output_utils import *

"""
Args json
- data_path:
- model_name:
"""


def main():
    print("@@@@@@@@@@@@@@@@ Code Started @@@@@@@@@@@@@@@@")
    
    
    ################# Load Data #################
    data_path = "Dataset/data_run_offset.npy"
    data = data_load(data_path)
    x, label = dataset_packing(data)
    print(x.shape)
    
    ################# Load Model #################
    model_name = "unetv2_rgbmse"
    model = load_model_(model_name, summary=True, space='local')
    generated_frames = predict(model, x, label)
    print(generated_frames.shape)
    
    ######## Save/Send/Visualization Data ########
    show_result(x, generated_frames, n=10)
    
    
if __name__ == "__main__":
    main()
    
    
    