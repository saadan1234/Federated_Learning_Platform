from datasets import load_dataset

# Function to load and display details of a dataset
def load_data(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    print(f"Type of Dataset: {type(dataset['train'][1])}")
    print(f"Length of Dataset: {len(dataset)}")
    return dataset

# Example Usage
dataset = load_data('uoft-cs/cifar10')