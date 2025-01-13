# Run below command for dataset generation and federated functionality
# ! pip install -q "flwr-datasets[vision]"

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Set the partitioner to create 10 partitions of the dataset
partitioner = IidPartitioner(num_partitions=10)

# Enter the name of any HuggingFace Dataset, Test set is not partitioned for evaluation across network
fds = FederatedDataset(
    dataset="uoft-cs/cifar10", partitioners={"train": partitioner}
)

# Load the first partition of the "train" split
partition = fds.load_partition(0, "train")

# You can access the whole "test" split of the base dataset (it hasn't been partitioned)
centralized_dataset = fds.load_split("test")