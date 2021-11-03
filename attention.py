import gym
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as transforms


class SelfAttention(nn.Module):
    """A simple self-attention solution."""

    def __init__(self, data_dim, dim_q):
        super(SelfAttention, self).__init__()
        self._layers = []

        self._fc_q = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_q)
        self._fc_k = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_k)

    def forward(self, input_data):
        # Expect input_data to be of shape (b, t, k).
        b, t, k = input_data.size()

        # Linear transforms.
        queries = self._fc_q(input=input_data)  # (b, t, q)
        keys = self._fc_k(input=input_data)  # (b, t, q)

        # Attention matrix.
        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b, t, t)
        scaled_dot = torch.div(dot, torch.sqrt(torch.tensor(k).float()))
        return scaled_dot


class FCStack(nn.Module):
    """Fully connected layers."""

    def __init__(self, input_dim, num_units, activation, output_dim):
        super(FCStack, self).__init__()
        self._activation = activation
        self._layers = []
        dim_in = input_dim
        for i, n in enumerate(num_units):
            layer = nn.Linear(dim_in, n)
            self._layers.append(layer)
            setattr(self, "_fc{}".format(i + 1), layer)  # Why though?
            dim_in = n
        output_layer = nn.Linear(dim_in, output_dim)
        self._layers.append(output_layer)

    def forward(self, input_data):
        x_input = input_data
        for layer in self._layers[:-1]:
            x_output = layer(x_input)
            if self._activation == "tanh":
                x_input = torch.tanh(x_output)
            elif self._activation == "elu":
                x_input = F.elu(x_output)
            else:
                x_input = F.relu(x_output)
        x_output = self._layers[-1](x_input)
        return x_output


class LSTMStack(nn.Module):
    """LSTM layers."""

    def __init__(self, input_dim, num_units, output_dim):
        super(LSTMStack, self).__init__()
        self._layers = []
        self._hidden_layers = len(num_units) if len(num_units) else 1
        self._hidden_size = num_units[0] if len(num_units) else output_dim
        self._hidden = (
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
        )
        if len(num_units):
            self._lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self._hidden_size,
                num_layers=self._hidden_layers,
            )
            self._layers.append(self._lstm)
            fc = nn.Linear(
                in_features=self._hidden_size,
                out_features=output_dim,
            )
            self._layers.append(fc)
        else:
            self._lstm = nn.LSTMCell(
                input_size=input_dim,
                hidden_size=self._hidden_size,
            )
            self._layers.append(self._lstm)

    def forward(self, input_data):
        x_input = input_data
        x_output, self._hidden = self._layers[0](x_input.view(1, 1, -1), self._hidden)
        x_output = torch.flatten(x_output, start_dim=0, end_dim=-1)
        if len(self._layers) > 1:
            x_output = self._layers[-1](x_output)
        return x_output

    def reset(self):
        self._hidden = (
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
        )


class Model:
    """Model with an underlying MLP or LSTM"""

    def __init__(
        self,
        input_dim,
        num_hiddens,
        activation,
        output_dim,
        output_activation,
        use_lstm,
        l2_coefficient,
    ):
        self._use_lstm = use_lstm
        self._output_dim = output_dim
        self._output_activation = output_activation
        if "roulette" in self._output_activation:
            assert self._output_dim == 1
            self._n_grid = int(self._output_activation.split("_")[-1])
            self._theta_per_grid = 2 * np.pi / self._n_grid
        self._l2_coefficient = abs(l2_coefficient)
        if self._use_lstm:
            self._fc_stack = LSTMStack(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=num_hiddens,
            )
        else:
            self._fc_stack = FCStack(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=num_hiddens,
                activation=activation,
            )
        self._layers = self._fc_stack.layers

    def output(self, inputs):
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy(inputs).float()
            fc_output = self._fc_stack(inputs)

            if self._output_activation == "tanh":
                output = torch.tanh(fc_output).squeeze().numpy()
            elif self._output_activation == "softmax":
                output = F.softmax(fc_output, dim=-1).squeeze().numpy()
            else:
                output = fc_output.squeeze().numpy()

            return output

    def reset(self):
        if hasattr(self._fc_stack, "reset"):
            self._fc_stack.reset()


class Agent:
    def __init__(
        self,
        image_size,
        query_dim,
        output_dim,
        output_activation,
        num_hiddens,
        l2_coefficient,
        patch_size,
        patch_stride,
        top_k,
        data_dim,
        activation,
        normalize_positions=False,
        use_lstm_controller=False,
    ):
        self._image_size = image_size
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._top_k = top_k
        self._l2_coefficient = l2_coefficient
        self._normalize_positions = normalize_positions

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self._patch_centers = self._patch_centers()

        self._attention = SelfAttention(
            data_dim=data_dim * self._patch_size ** 2,
            dim_q=query_dim,
        )

        self._model = Model(
            input_dim=self._top_k * 2,
            num_hiddens=num_hiddens,
            activation=activation,
            output_dim=output_dim,
            output_activation=output_activation,
            l2_coefficient=l2_coefficient,
            use_lstm=use_lstm_controller,
        )

    def _patch_centers(self):
        """Determines centers of patches based on image_size, patch_size and patch_stride"""
        # Images are squares, so patch rows == cols
        n = int((self.image_size - self.patch_size) / self.patch_stride + 1)
        offset = self._patch_size // 2  # Floor division
        patch_centers = []
        for i in range(n):
            patch_center_row = offset + i * self.patch_stride
            for j in range(n):
                patch_center_col = offset + j * self.patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        return torch.tensor(patch_centers).float()

    def get_output(self, inputs):
        with torch.no_grad():
            # ob.shape = (h, w, c)
            ob = self._transform(inputs).permute(1, 2, 0)
            # print(ob.shape)
            h, w, c = ob.size()
            patches = ob.unfold(0, self._patch_size, self._patch_stride).permute(0, 3, 1, 2)
            patches = patches.unfold(2, self._patch_size, self._patch_stride).permute(
                0, 2, 1, 4, 3
            )
            patches = patches.reshape((-1, self._patch_size, self._patch_size, c))

            # flattened_patches.shape = (1, n, p * p * c)
            flattened_patches = patches.reshape((1, -1, c * self._patch_size ** 2))
            # attention_matrix.shape = (1, n, n)
            attention_matrix = self._attention(flattened_patches)
            # patch_importance_matrix.shape = (n, n)
            patch_importance_matrix = torch.softmax(attention_matrix.squeeze(), dim=-1)
            # patch_importance.shape = (n,)
            patch_importance = patch_importance_matrix.sum(dim=0)
            # extract top k important patches
            ix = torch.argsort(patch_importance, descending=True)
            top_k_ix = ix[: self._top_k]

            centers = self._patch_centers[top_k_ix]

            centers = centers.flatten(0, -1)
            if self._normalize_positions:
                centers = centers / self._image_size

            return self._model.get_output(centers)

    def reset(self):
        self._model.reset()


if __name__ == "__main__":
    env = gym.make("CarRacing-v0")
