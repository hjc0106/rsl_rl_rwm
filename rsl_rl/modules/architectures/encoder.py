import re
import torch
from torch import nn
from rsl_rl.networks import ConvEncoder, WM_MLP

class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
        use_camera = False,
    ):
        super(MultiEncoder, self).__init__()
        self.use_camera = use_camera
        excluded = ("is_first", "is_last", "is_terminal", "reward",  "height_map")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
            print('cnn outdim', self._cnn.outdim)
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = WM_MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units
            print('mlp outdim', mlp_units)

        print('total outdim:', self.outdim)

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            if(self.use_camera):
                inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
                outputs.append(self._cnn(inputs))
            else:
                outputs.append(torch.zeros((obs["is_first"].shape + (self._cnn.outdim,)), device=obs["is_first"].device))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs
