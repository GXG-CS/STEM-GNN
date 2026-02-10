from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
from utils.others import mask2idx


def get_loader(data, split, labels, params):
    task = params['task']
    assert params["setting"] == "standard", "Only standard setting is supported"

    if task == "node":
        train_loader = NeighborLoader(
            data,
            num_neighbors=[10] * params["num_layers"],
            input_nodes=mask2idx(split["train"]),
            batch_size=params["batch_size"],
            num_workers=8,
            shuffle=True,
        )
        subgraph_loader = NeighborLoader(
            data,
            num_neighbors=[-1] * params["num_layers"],
            batch_size=512,
            num_workers=8,
            shuffle=False,
        )
        return train_loader, subgraph_loader

    elif task == "link":
        train_loader = LinkNeighborLoader(
            data,
            num_neighbors=[30] * params["num_layers"],
            edge_label_index=data.edge_index[:, split["train"]],
            edge_label=labels[split["train"]],
            batch_size=params["batch_size"],
            num_workers=8,
            shuffle=True,
        )
        subgraph_loader = LinkNeighborLoader(
            data,
            num_neighbors=[-1] * params["num_layers"],
            edge_label_index=data.edge_index,
            edge_label=labels,
            batch_size=4096,
            num_workers=8,
            shuffle=False,
        )
        return train_loader, subgraph_loader

    elif task == "graph":
        train_dataset = data[split["train"]]
        val_dataset = data[split["valid"]]
        test_dataset = data[split["test"]]

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=8,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=8,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=8,
        )

        return train_loader, val_loader, test_loader
