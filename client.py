from flwr.client import Client, ClientApp, NumPyClient
from typing import Dict, OrderedDict
from flwr.common import NDArrays, Scalar
import torch
from flwr.common import Context
from helpers import SimpleModel, evaluate_model, train_model
from data import train_sets, testset


# Function to update the weights of the local model
# net : local model, parameters: updates weights
def set_weights(net, parameters):
    # params_dict : tuple with layer name, respective weights
    params_dict = zip(net.state_dict().keys(), parameters)
    # convert v : weights into tensor with respective key k
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    # updates the net: models weights
    net.load_state_dict(state_dict, strict=True)

# Function to get weights of the local model
# net : local model
def get_weights(net):
    # Moves tensors to memory and then converts it into Numpy array
    ndarray = [
        # Keys are ignored and only values are used.
        val.cpu().numpy() for _, val in net.state_dict().items()  
    ]
    # returns the array with the model weights
    return ndarray


# Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset):
        self.net = net
        self.trainset = trainset
        self.testset = testset

    # Function to train the local model
    # parameters: global model weights, config: additional hyperparameters
    def fit(self, parameters, config):
        # Set weights of the local model
        set_weights(self.net, parameters)
        # Train the local model
        train_model(self.net, self.trainset)
        # Returns the updates weights, number of training examples, additional dict for metadeta/metrics
        return get_weights(self.net), len(self.trainset), {}
    
    # Function to evaluate global model weights performance on local data
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Set weights of the local model
        set_weights(self.net, parameters)
        # Return accuracy and loss of the model on local test data
        loss, accuracy = evaluate_model(self.net, self.testset)
        return loss, len(self.testset), {"accuracy": accuracy}

# Client Function that creats individual client instances.
def client_fn(context: Context)-> Client:
    # The "Context" contains the client id, config settings, state and run file.
    net = SimpleModel()
    # The following id gets the partition corresponding to client 
    partition_id = int(context.node_config["partition-id"])
    # The data assigned to the client is based on its partition id
    client_train = train_sets[int(partition_id)]
    # Contains the overall test set to evaluate the client node.
    client_test = testset
    # Returns the flower client with its corresponding train and test set.
    return FlowerClient(net, client_train, client_test).to_client