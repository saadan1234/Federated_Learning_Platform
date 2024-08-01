from logging import INFO, log
from client import client_fn, get_weights, set_weights
from helpers import SimpleModel, compute_confusion_matrix, evaluate_model, fit_config, plot_confusion_matrix, backend_setup
from data import testset, testset_137, testset_258, testset_469
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig
from flwr.server import ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.client import ClientApp

# Client app with the initial configuration of the client
client = ClientApp(client_fn)

# Evaluation of the server aggragates model using the local dataset.
def evaluate(server_round, parameters, config):
    net = SimpleModel()
    # Set weights of the local model for testing
    set_weights(net, parameters)

    # Ignores the loss and returns the accuracy to evaluate the model performance.
    _, accuracy = evaluate_model(net, testset)
    _, accuracy137 = evaluate_model(net, testset_137) 
    _, accuracy258 = evaluate_model(net, testset_258)
    _, accuracy469 = evaluate_model(net, testset_469)

    log(INFO, "test accuracy on all digits: %.4f", accuracy)
    log(INFO, "test accuracy on [1,3,7]: %.4f", accuracy137)
    log(INFO, "test accuracy on [2,5,8]: %.4f", accuracy258)
    log(INFO, "test accuracy on [4,6,9]: %.4f", accuracy469)

    # Complete the training after 3 server rounds.
    if server_round == 3:
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model")

net = SimpleModel()
# Get the final model parameters and pass on the weights in the form or numpy array
params = ndarrays_to_parameters(get_weights(net))

# Server function to define the server id, config settings and state
def server_fn(context: Context):
    # FedAvg to simple take average of the all the client weights
    strategy = FedAvg(
        # Train the local model on 100% local data available to the client. (1 = 100%)
        fraction_fit=1.0,
        # Evaluate the local model on 0% of the local data available to the cleint. (0 = 0%)
        fraction_evaluate=0.0,
        # Intitialize the server model with the given below weights (or hyperparameters)
        initial_parameters=params,
        # Evaluate the global model based on the evaluate function
        evaluate_fn=evaluate,
        # Training configurations.
        on_fit_config_fn=fit_config,
    )
    # Server training details.
    config=ServerConfig(num_rounds=3)
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )

# Create the server app given the configuration settings in the server function
server = ServerApp(server_fn=server_fn)

# Initiate the simulation passing the server and client apps
# Specify the number of super nodes that will be selected on every round
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=3,
    backend_config=backend_setup,
)