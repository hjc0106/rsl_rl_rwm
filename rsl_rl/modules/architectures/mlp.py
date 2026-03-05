import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd
import numpy as np
from rsl_rl.utils import tools


class MLPBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        
        base_shape = architecture_config["base_shape"]
        layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in base_shape:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.layers = nn.Sequential(*layers).to(self.device)
        self.layers.train()
        
    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1).flatten(1, 2)
        x = self.layers(x)
        return x
    
    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass


class MLPStateHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device
        self.state_mean_shape = architecture_config["state_mean_shape"]
        self.state_logstd_shape = architecture_config["state_logstd_shape"]

        state_mean_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.state_mean_shape:
            state_mean_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            state_mean_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        state_mean_layers.append(nn.Linear(self.state_mean_shape[-1], state_dim))
        self.state_mean_layers = nn.Sequential(*state_mean_layers).to(self.device)
        self.state_mean_layers.train()

        if self.state_logstd_shape is not None:
            self.output_std = True
            state_logstd_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in self.state_logstd_shape:
                state_logstd_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                state_logstd_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            state_logstd_layers.append(nn.Linear(self.state_logstd_shape[-1], state_dim))
            self.state_logstd_layers = nn.Sequential(*state_logstd_layers).to(self.device)
            self.state_logstd_layers.train()
        else:
            self.output_std = False

        if self.output_std:
            self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * -5.0)
            self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * 0.0)

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
            x_state_batch = x_state_batch.flatten(0, 1).unsqueeze(1)
        else:
            sequence_len = 0
        state_mean = self.state_mean_layers(x) + x_state_batch[:, -1]
        state_logstd = self.state_logstd_layers(x) if self.output_std else -torch.inf * torch.ones(x.shape[0], self.state_dim, device=self.device)
        if self.output_std:
            self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            state_logstd = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - state_logstd)
            state_logstd = self.state_min_logstd + nn.functional.softplus(state_logstd - self.state_min_logstd)
        if sequence_len > 0:
            state_mean = state_mean.view(-1, sequence_len, self.state_dim)
            state_logstd = state_logstd.view(-1, sequence_len, self.state_dim)
        return state_mean, torch.exp(state_logstd)

    def reset(self):
        pass
    
    def reset_partial(self, batch_indices):
        pass


class MLPAuxiliaryHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        architecture_config: dict = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device

        if extension_dim > 0:
            extension_shape = architecture_config["extension_shape"]
            extension_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in extension_shape:
                extension_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                extension_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            extension_layers.append(nn.Linear(extension_shape[-1], extension_dim))
            self.extension_layers = nn.Sequential(*extension_layers).to(self.device)
            self.extension_layers.train()

        if contact_dim > 0:
            contact_shape = architecture_config["contact_shape"]
            contact_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in contact_shape:
                contact_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                contact_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            contact_layers.append(nn.Linear(contact_shape[-1], contact_dim))
            self.contact_layers = nn.Sequential(*contact_layers).to(self.device)
            self.contact_layers.train()
        
        if termination_dim > 0:
            termination_shape = architecture_config["termination_shape"]
            termination_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in termination_shape:
                termination_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                termination_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            termination_layers.append(nn.Linear(termination_shape[-1], termination_dim))
            self.termination_layers = nn.Sequential(*termination_layers).to(self.device)
            self.termination_layers.train()

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
        else:
            sequence_len = 0
        
        extension_pred = self.extension_layers(x) if self.extension_dim > 0 else None
        contact_logits = self.contact_layers(x) if self.contact_dim > 0 else None
        termination_logits = self.termination_layers(x) if self.termination_dim > 0 else None
        
        if sequence_len > 0:
            extension_pred = extension_pred.view(-1, sequence_len, self.extension_dim) if self.extension_dim > 0 else None
            contact_logits = contact_logits.view(-1, sequence_len, self.contact_dim) if self.contact_dim > 0 else None
            termination_logits = termination_logits.view(-1, sequence_len, self.termination_dim) if self.termination_dim > 0 else None
        
        return extension_pred, contact_logits, termination_logits

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass

class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist