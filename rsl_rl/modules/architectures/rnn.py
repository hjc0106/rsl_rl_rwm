import torch
import torch.nn as nn
from .rssm import RSSMBase


class RNNBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        
        rnn_type = architecture_config["rnn_type"]
        rnn_num_layers = architecture_config["rnn_num_layers"]
        rnn_hidden_size = architecture_config["rnn_hidden_size"]
        self.memory = Memory(input_dim, device, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        
    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)
        x = self.memory(x)
        return x
    
    def reset(self):
        self.memory.reset()
        
    def reset_partial(self, batch_indices):
        self.memory.reset_partial(batch_indices)


class Memory(nn.Module):
    def __init__(self, input_dim: int, device: str, type: str, num_layers: int, hidden_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=self.input_dim, hidden_size=hidden_size, num_layers=num_layers, device=self.device, batch_first=True)
        self.hidden_states = None

    def forward(self, x):
        x, self.hidden_states = self.rnn(x, self.hidden_states)
        return x[:, -1]
    
    def reset(self):
        self.hidden_states = None

    def reset_partial(self, batch_indices):
        if self.hidden_states is not None:
            self.hidden_states[:, batch_indices] = 0.0


class RSSMDynamicsBase(nn.Module):
    """Stateful wrapper around RSSMBase that provides the same forward(x_state_batch, x_action_batch)
    interface as RNNBase, making it a drop-in replacement in SystemDynamicsEnsemble."""

    def __init__(self, input_dim: int, device: str, architecture_config: dict = None):
        super().__init__()
        self.device = device
        self.rssm = RSSMBase(
            input_dim=input_dim,
            device=device,
            architecture_config=architecture_config,
        )
        # KL loss hyperparams (read from config with sensible defaults)
        self._kl_free = architecture_config.get("kl_free", 1.0)
        self._kl_dyn_scale = architecture_config.get("kl_dyn_scale", 0.5)
        self._kl_rep_scale = architecture_config.get("kl_rep_scale", 0.1)

        self._prev_state = None
        self._prev_action = None
        self._kl_loss_val = None

    def imagine(self, x_state_batch: torch.Tensor, x_action_batch: torch.Tensor):
        # TODO: multiple steps
        prior = self.rssm.imagine_with_action(x_action_batch, x_state_batch)
        return self.rssm.get_feat(prior)


    def forward(self, x_state_batch: torch.Tensor, x_action_batch: torch.Tensor) -> torch.Tensor:
        """Process state/action sequences through the RSSM.

        Args:
            x_state_batch:  (batch, seq_len, state_dim)  — treated as the observation embedding.
            x_action_batch: (batch, seq_len, action_dim)

        Returns:
            features: (batch, stoch_feat_dim + deter_dim)  via rssm.get_feat(post)
        """
        seq_len = x_state_batch.shape[1]

        if seq_len > 1:
            # History warmup: process the full context window with observe().
            # is_first is 1 for the very first timestep, 0 for the rest.
            batch_size = x_state_batch.shape[0]
            is_first = torch.zeros(batch_size, seq_len, device=self.device)
            is_first[:, 0] = 1.0

            post, prior = self.rssm.observe(x_state_batch, x_action_batch, is_first)
            # post values have shape (batch, seq_len, ...) — take the last timestep
            last_post = {k: v[:, -1] for k, v in post.items()}
            last_prior = {k: v[:, -1] for k, v in prior.items()}

            kl_raw = self.rssm.kl_loss(
                post, prior,
                free=self._kl_free,
                dyn_scale=self._kl_dyn_scale,
                rep_scale=self._kl_rep_scale,
            )
            self._kl_loss_val = kl_raw[0].mean()

            self._prev_state = last_post
            self._prev_action = x_action_batch[:, -1]
            return self.rssm.get_feat(last_post)
        else:
            # Single-step rollout: posterior update with obs_step.
            embed = x_state_batch[:, 0]        # (batch, state_dim)
            action = x_action_batch[:, 0]      # (batch, action_dim)

            is_first = torch.zeros(embed.shape[0], device=self.device)
            if self._prev_state is None:
                is_first = torch.ones(embed.shape[0], device=self.device)

            post, prior = self.rssm.obs_step(
                self._prev_state, self._prev_action, embed, is_first, sample=True
            )

            kl_raw = self.rssm.kl_loss(
                post, prior,
                free=self._kl_free,
                dyn_scale=self._kl_dyn_scale,
                rep_scale=self._kl_rep_scale,
            )
            self._kl_loss_val = kl_raw[0].mean()

            self._prev_state = post
            self._prev_action = action
            return self.rssm.get_feat(post)

    @property
    def kl_loss(self) -> torch.Tensor:
        """Return the KL loss computed during the most recent forward pass."""
        if self._kl_loss_val is None:
            return torch.tensor(0.0, device=self.device)
        return self._kl_loss_val

    def reset(self):
        self._prev_state = None
        self._prev_action = None
        self._kl_loss_val = None

    def reset_partial(self, batch_indices):
        if self._prev_state is None:
            return
        is_first = torch.zeros(
            list(self._prev_state.values())[0].shape[0], device=self.device
        )
        is_first[batch_indices] = 1.0
        init_state = self.rssm.initial(list(self._prev_state.values())[0].shape[0])
        for key, val in self._prev_state.items():
            is_first_r = is_first.reshape(
                is_first.shape + (1,) * (len(val.shape) - len(is_first.shape))
            )
            self._prev_state[key] = (
                val * (1.0 - is_first_r) + init_state[key] * is_first_r
            )
        if self._prev_action is not None:
            self._prev_action[batch_indices] = 0.0
