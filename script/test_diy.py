import torch_geometric
import torchvision
from torchdrug import core
import sys
import os

from gearnet import model, cdconv, gvp, dataset, task, protbert
from torchdrug.data.dataset import ProteinDataset
from torchdrug.data.dataloader import graph_collate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util

def test_diy():
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    task = core.Configurable.load_config_dict(cfg.task)
    task.preprocess(dataset, None, None)
    cfg.task.task = [_ for _ in range(len(dataset.tasks))]
    batch = graph_collate([dataset[i] for i in [1, 4]])

    model = core.Configurable.load_config_dict(cfg.task)
    graph = batch["graph"]
    graph = task.graph_construction_model(graph)
    res = model.model(graph, graph.node_feature.float())
    assert graph.node_position.shape[0] == res["node_feature"].shape[0]
    assert res["node_feature"].shape[1] == 4352
    print("Finished!!")

if __name__ == "__main__":
    test_diy()
