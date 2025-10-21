import flwr as fl

# ✅ Weighted average for federated metric aggregation
def weighted_average(metrics):
    """
    metrics: list of tuples like [(metrics_dict, num_examples), ...]
    We return a dict instead of a float (required by Flower).
    """
    total_examples = 0
    weighted_acc_sum = 0.0

    for m, n in metrics:
        # Handle case where num_examples is dict
        if isinstance(n, dict):
            n = list(n.values())[0]

        # Extract accuracy safely
        if isinstance(m, dict):
            if "accuracy" in m:
                weighted_acc_sum += m["accuracy"] * n
            elif "acc" in m:
                weighted_acc_sum += m["acc"] * n

        total_examples += n

    if total_examples == 0:
        return {"accuracy": 0.0}

    avg_acc = weighted_acc_sum / total_examples
    return {"accuracy": avg_acc}  # ✅ Return dict, not float


# ✅ Start Flower server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    ),
)


