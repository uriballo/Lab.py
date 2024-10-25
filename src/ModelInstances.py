import pennylane as qml
from pennylane import numpy as np
import src.Models

def choose_model(model_string, n_classes):
    if model_string == "QNN4ESAT":
        return QNN4ESAT(n_classes)
    else:
        raise ValueError(f"Unknown model: {model_string}")

def QNN4ESAT(n_classes):
    n_qubits = 8
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights):
        qml.broadcast(unitary=qml.Hadamard, wires=range(n_qubits), pattern="single")
        features = inputs * 2 * np.pi
        qml.AngleEmbedding(features=features, wires=range(n_qubits), rotation='Z')

        for W in weights:
            qml.broadcast(unitary=qml.CNOT, wires=range(n_qubits), pattern="ring")

            for idx in range(n_qubits):
                k = idx * 3
                
                qml.RY(W[k], wires=idx)
                qml.RX(W[k + 1], wires=idx)
                qml.RZ(W[k + 2], wires=idx)
                
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    
    repetitions = 2
    weight_shapes = {'weights': (repetitions, 3 * n_qubits)}
    
    return src.Models.QNN4ESAT(circuit, weight_shapes, n_qubits, "n", n_classes)
    