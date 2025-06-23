"""
Run tests for Network class on MLP functionality only
--- Written by Chat GPT ---
"""

import numpy as np
from net.tensor import Tensor
from net.blocks.mlp import MLP
from net.network import Network
from net.train import sgd

def make_toy_data(n=20, d_in=4, d_out=3, seed=1):
    np.random.seed(seed)
    X = Tensor(np.random.randn(n, d_in), grad_required=True)
    y = Tensor(np.random.randint(0, d_out, size=n), grad_required=False)
    return X, y

def test_forward_output_shape():
    X, y = make_toy_data()
    mlp = MLP(size=(4, 10, 3), act='relu')
    net = Network(blocks=(mlp,), training=True)
    out = net.forward(X, y)
    assert isinstance(out, Tensor)
    assert out.shape == (), "Loss should be scalar"

def test_prediction_shape_and_range():
    X, y = make_toy_data()
    mlp = MLP(size=(4, 8, 3), act='tanh')
    net = Network(blocks=(mlp,), training=False)
    net.forward(X)  # must be called first to compute .out
    preds = net.preds()
    assert isinstance(preds, Tensor)
    assert preds.shape == (20,)
    assert np.all((preds >= 0) & (preds < 3)), "Predicted class indices must be in valid range"

def test_loss_computation():
    X, y = make_toy_data()
    mlp = MLP(size=(4, 5, 3), act='relu')
    net = Network(blocks=(mlp,), training=True)
    loss = net.forward(X, y)
    assert isinstance(loss, Tensor)
    assert loss.shape == (), "Loss should be a scalar"

def test_backward_pass_runs():
    X, y = make_toy_data()
    mlp = MLP(size=(4, 6, 3), act='tanh')
    net = Network(blocks=(mlp,), training=True)
    loss = net.forward(X, y)
    loss.zero_grad_deep()
    loss.backward()
    for param in net.parameters():
        assert param.grad is not None, "Gradient should not be None after backward"
        assert param.grad.shape == param.shape, "Gradient shape mismatch"

def test_sgd_training_decreases_loss():
    X, y = make_toy_data(n=100)
    mlp = MLP(size=(4, 10, 3), act='relu')
    net = Network(blocks=(mlp,), training=True)

    initial_loss = net.forward(X, y).value
    sgd(net, X, y, lr=0.1, batch_size=10, steps=100)
    final_loss = net.forward(X, y).value

    assert final_loss < initial_loss, f"Expected loss to decrease: {initial_loss} → {final_loss}"

def test_mlp_learns_xor():
    # XOR truth table
    X_raw = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_raw = np.array([0, 1, 1, 0])  # XOR labels

    # Wrap as Tensor objects
    X = Tensor(X_raw, grad_required=True)
    y = Tensor(y_raw, grad_required=False)

    # Create a small MLP: input 2 → hidden 4 → output 2 (binary classification)
    mlp = MLP(size=(2, 4, 2), act='tanh')
    net = Network(blocks=(mlp,), training=True)

    # Train using SGD
    sgd(net, X, y, lr=0.1, batch_size=4, steps=500)

    # Evaluate accuracy
    net.training = False
    net.forward(X)
    preds = net.preds()

    acc = np.mean(preds == y)
    assert acc >= 0.75, f"XOR accuracy too low: {acc}"
    
def trace_graph(tensor, visited=None, indent=0):
    if visited is None:
        visited = set()
    if id(tensor) in visited:
        return
    visited.add(id(tensor))

    print("  " * indent + f"Tensor(id={id(tensor)}): op='{tensor.op}', shape={tensor.shape}")
    if tensor.grad is not None:
        print("  " * indent + f"   grad.shape = {tensor.grad.shape}")
    if hasattr(tensor, "parents"):
        for parent in tensor.parents:
            trace_graph(parent, visited, indent + 1)

def test_rnn_manual_trace():
    np.random.seed(0)

    X, y = make_toy_data()
    mlp = MLP(size=(4, 6, 3), act='tanh')
    net = Network(blocks=(mlp,), training=True)
    loss = net.forward(X, y)
    loss.zero_grad_deep()
    loss.backward()

    print("\n--- Computation Graph Trace ---")
    trace_graph(loss)

    print("\n--- Parameter Gradients ---")
    for p in mlp.parameters():
        print(f"{p.op}: grad =\n{p.grad}")

if __name__ == "__main__":
    test_rnn_manual_trace()