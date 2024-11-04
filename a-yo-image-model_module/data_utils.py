import numpy as np

def label_to_one_hot(labels, num_classes, repeat):
    """
    Summarize: This function returns one-hot vectors for each frame.
    Input: 
        - Shape:
        - Type: 
        - Example: 
    Output:
        - Shape: 
        - Type: 
        - Example: 
    """
    current_label_dataset = np.zeros((num_classes))
    current_label_dataset[labels]  = 1
    current_label_dataset = np.expand_dims(current_label_dataset, axis=0)
    current_label_dataset = np.repeat(current_label_dataset, repeats=repeat, axis=0)
    return current_label_dataset

def data_load(data_path):
    """
    Summarize: This function returns one-hot vectors for each frame.
    Input: 
        - Shape:
        - Type: 
        - Example: 
    Output:
        - Shape: 
        - Type: 
        - Example: 
    """
    dataset = np.load(data_path)
    
    print("Successfully loaded the dataset")
    return dataset

def dataset_packing(dataset):
    """
    Summarize: This function prepares dataset to fit the input shape for the model.
    Input: 
        - Shape:
        - Type: 
        - Example: 
    Output:
        - Shape: 
        - Type: 
        - Example: 
    """
    
    x_dataset = dataset[:,0,:,:,:]
    rr  = x_dataset.shape[0]
    label_dataset = label_to_one_hot(1,10, rr)

    for i in range(8) :
        x_dataset = np.concatenate((x_dataset, dataset[:,0,:,:,:] ), axis = 0)
        label_dataset = np.concatenate((label_dataset, label_to_one_hot(i + 2,10, rr)), axis = 0)

    x_train = x_dataset[0]
    x_train = np.expand_dims(x_train, axis=0)
    
    label_train = label_dataset[0]
    label_train = np.expand_dims(label_train, axis=0)
    

    return x_train, label_train

