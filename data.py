# Run below command for dataset generation and federated functionality
# ! pip install -q "flwr-datasets[vision]"

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.partitioner import PathologicalPartitioner

# User will enter dataset name, partitioner (it can be based on label, or skewness or by default IID partitioning),

def dataset_initialization(dataset_name="uoft-cs/cifar10", num_partitions=1, num_classes_per_partition=None, alpha=None, visualize=False):
    if num_classes_per_partition is not None:
        partitioner = PathologicalPartitioner(num_partitions=num_partitions, partition_by="label", num_classes_per_partition=num_classes_per_partition)
    elif alpha is not None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=alpha, seed=42, min_partition_size=0)
    else:
        partitioner = IidPartitioner(num_partitions=num_partitions)
    
    fds = FederatedDataset(
        dataset= dataset_name, partitioners={"train": partitioner}
        )
    
    partitioner = fds.partitioners["train"]

    if visualize:
      fig, ax, df = plot_label_distributions(
        partitioner,
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
      )

    return fds, partitioner

def get_partition_params():
    """Prompt user for dataset partitioning parameters."""
    dataset_name = input("Enter dataset name: ").strip()
    num_partitions = int(input("Enter the number of partitions (positive integer): "))
    choice = input("Partition by 'label' or 'skewness': ").strip().lower()
    params = {"dataset_name": dataset_name, "num_partitions": num_partitions, "num_classes_per_partition":None, "alpha":None}
    if choice == "label":
        params["num_classes_per_partition"] = int(input("Enter number of classes per partition (positive integer): "))
    elif choice == "skewness":
        params["alpha"] = float(input("Enter alpha value (positive float): "))
    else:
        raise ValueError("Invalid choice! Must be 'label' or 'skewness'.")
    return params

params = get_partition_params()
dataset=dataset_initialization(dataset_name=params["dataset_name"], num_partitions=params["num_partitions"], num_classes_per_partition=params["num_classes_per_partition"], alpha=params["alpha"], visualize=True)