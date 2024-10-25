import torch
import torch.nn as nn
import pennylane as qml

class QuantumLinear(nn.Module):
    def __init__(self, in_features, out_features, q_circuit, weight_shapes, num_qubits, embedding="n"):
        """
        :param in_features: number of input features.
        :param out_features: number of output features (currently not used).
        :param q_circuit: quantum circuit used to process the input features.
        :param weight_shapes: shape of the quantum circuit weights.
        :param num_qubits: number of qubits of the quantum circuit.
        :param embedding: either `n` or `2n`, depending on the embedding used. Use `n` for embeddings like angle and
                            `2n` for amplitude.
        """
        super(QuantumLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features  # Not used, currently depends on the embedding.
        self.num_qubits = num_qubits
        self.embedding = embedding
        
        if embedding == "n":
            assert in_features % num_qubits == 0, ("The number of input features should be divisible by the number of "
                                                   "qubits")
            self.n_circuits = in_features // num_qubits
            self.circuits = nn.ModuleList([
                qml.qnn.TorchLayer(q_circuit, weight_shapes) for _ in range(self.n_circuits)
            ])
        elif embedding == "2n":
            self.n_circuits = in_features // 2 ** num_qubits
            self.circuits = nn.ModuleList([
                qml.qnn.TorchLayer(q_circuit, weight_shapes) for _ in range(self.n_circuits)
            ])
        else:
            ValueError("Embedding not recognized use either `n` or `2n`.")
            
        for circ in self.circuits:
            nn.init.xavier_uniform_(circ.weights)

    def forward(self, x):
        x = x.view(x.shape[0], self.n_circuits, self.num_qubits)
        x = torch.cat([circ(x[:, i, :]) for i, circ in enumerate(self.circuits)], dim=1)
        return x


class QuConv2D_MQ(nn.Module):
    def __init__(self, kernel_size, stride, in_channels, out_channels, q_circuit, weight_shapes, num_qubits,
                 embedding="n", pad=False, padding=0, dilation=1, padding_mode='constant'):
        """
        :param kernel_size: size of the convolution window, kernel_size · kernel_size.
        :param stride: stride of the convolution window.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels, currently this depends on the number of qubits.
        :param q_circuit: quantum circuit used to process the window.
        :param weight_shapes: shape of the quantum's circuit weights.
        :param num_qubits: number of qubits of the quantum circuit.
        :param embedding: either `n` or `2n`, depending on the embedding used. Use `n` for embeddings like angle and
                            `2n` for amplitude.
        :param pad: True if padding should be used.
        :param padding: padding value added to the input image.
        :param dilation: dilation of the convolution window.
        :param padding_mode: type of padding. Should be 'constant', 'reflect', 'replicate', or 'circular'.
        """
        super(QuConv2D_MQ, self).__init__()

        if embedding == "n":
            assert kernel_size * kernel_size <= num_qubits, ("Convolution window is too big for the chosen circuit "
                                                             "and embedding")
        if embedding == "2n":
            assert kernel_size * kernel_size <= 2 ** num_qubits, ("Convolution window is too big for "
                                                                  "the chosen circuit and embedding")

        #assert out_channels == num_qubits * in_channels, ("For now, the number of output channels must be num_qubits * "
        #                                                  "in_channels")

        assert padding_mode in ['constant', 'reflect', 'replicate', 'circular'], ("Wrong padding mode. Should be "
                                                                                  "'constant', 'reflect', "
                                                                                  "'replicate', or 'circular'.")

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_qubits = num_qubits
        self.embedding = embedding
        self.use_padding = pad
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode

        self.quanv = nn.ModuleList([
            qml.qnn.TorchLayer(q_circuit, weight_shapes) for _ in range(in_channels)
        ])

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride
        pad = self.padding
        dil = self.dilation

        # Add padding to the input tensor
        if self.use_padding:
            x = nn.functional.pad(x, (pad, pad, pad, pad), mode=self.padding_mode)

        # Compute output dimensions considering padding and dilation
        out_height = (height + 2 * pad - dil * (kh - 1) - 1) // sh + 1
        out_width = (width + 2 * pad - dil * (kw - 1) - 1) // sw + 1

        # Initialize output tensor to store outputs from all qubits
        out = torch.zeros((batch_size, channels, out_height, out_width, self.num_qubits))

        # Apply quantum convolution
        for i in range(channels):  # For each channel
            for h in range(out_height):
                for w in range(out_width):
                    x_start, x_end = h * sh, h * sh + dil * (kh - 1) + 1
                    y_start, y_end = w * sw, w * sw + dil * (kw - 1) + 1
                    patch = x[:, i, x_start:x_end:dil, y_start:y_end:dil].reshape(batch_size, -1)
                    patch = nn.functional.normalize(patch)
                    quanv_output = self.quanv[i](patch)
                    out[:, i, h, w, :] = quanv_output  # Store all qubit outputs

        return out.reshape(batch_size, self.out_channels, out_height, out_width)


class QuConv2D_MC(nn.Module):
    def __init__(self, kernel_size, stride, in_channels, out_channels, q_circuit, weight_shapes, num_qubits,
                 embedding="n", pad=False, padding=0, dilation=1, padding_mode='constant'):
        """
        :param kernel_size: size of the convolution window, kernel_size · kernel_size.
        :param stride: stride of the convolution window.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels, currently this depends on the number of qubits.
        :param q_circuit: quantum circuit used to process the window.
        :param weight_shapes: shape of the quantum's circuit weights.
        :param num_qubits: number of qubits of the quantum circuit.
        :param embedding: either n or 2n, depending on the embedding used. Use n for embeddings like angle and
                            2n for amplitude.
        :param pad: True if padding should be used.
        :param padding: padding added to the input image.
        :param dilation: dilation of the convolution window.
        :param padding_mode: type of padding. Should be 'constant', 'reflect', 'replicate', or 'circular'.
        """
        super(QuConv2D_MC, self).__init__()

        if embedding == "n":
            assert kernel_size * kernel_size <= num_qubits, ("Convolution window is too big for the chosen circuit "
                                                             "and embedding")
        if embedding == "2n":
            assert kernel_size * kernel_size <= 2 ** num_qubits, ("Convolution window is too big for "
                                                                  "the chosen circuit and embedding")

        assert padding_mode in ['constant', 'reflect', 'replicate', 'circular'], ("Wrong padding mode. Should be "
                                                                                  "'constant', 'reflect', "
                                                                                  "'replicate', or 'circular'.")
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_qubits = num_qubits
        self.embedding = embedding
        self.use_padding = pad
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode

        self.quanv = nn.ModuleList([
            qml.qnn.TorchLayer(q_circuit, weight_shapes) for _ in range(out_channels)
        ])

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride
        pad = self.padding
        dil = self.dilation

        # Add padding to the input tensor
        if self.use_padding:
            x = nn.functional.pad(x, (pad, pad, pad, pad), mode=self.padding_mode)

        # Compute output dimensions considering padding and dilation
        out_height = (height + 2 * pad - dil * (kh - 1) - 1) // sh + 1
        out_width = (width + 2 * pad - dil * (kw - 1) - 1) // sw + 1

        # Initialize output tensor to store outputs from all qubits
        out = torch.zeros((batch_size, self.out_channels, out_height, out_width))

        # Apply quantum convolution
        for i in range(self.out_channels):
            for h in range(out_height):
                for w in range(out_width):
                    x_start, x_end = h * sh, h * sh + dil * (kh - 1) + 1
                    y_start, y_end = w * sw, w * sw + dil * (kw - 1) + 1
                    patch = x[:, i % channels, x_start:x_end:dil, y_start:y_end:dil].reshape(batch_size, -1)
                    quanv_output = self.quanv[i](patch)[0]
                    out[:, i, h, w] = quanv_output

        return out.reshape(batch_size, self.out_channels, out_height, out_width)
