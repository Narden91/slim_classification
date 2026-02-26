SLIM_VERSIONS = ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]


def extract_model_size(model):
    if model is None:
        return 0

    if hasattr(model, "nodes_count"):
        return int(model.nodes_count)
    if hasattr(model, "node_count"):
        return int(model.node_count)
    if hasattr(model, "nodes"):
        return int(model.nodes)

    if hasattr(model, "collection"):
        nodes_collection = [getattr(tree, "nodes", 0) for tree in model.collection]
        return int(sum(nodes_collection) + max(0, len(model.collection) - 1))

    return 0


def extract_block_count(model):
    if model is None:
        return 0
    if hasattr(model, "collection"):
        return int(len(model.collection))
    return 1
