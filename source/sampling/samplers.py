import torch
import torch.nn as nn
import random
from torch.distributions.categorical import Categorical
import re

class Sampler:
    """
   Sampler base class
    """
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
         Sample from logits

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        """
        raise NotImplementedError()
    
class GreedySampler(Sampler):
    def __init__(self):
        self.softmax = nn.Softmax(dim = -1)
    def __call__(self, logits: torch.Tensor):
        """
        Sample the most likely token from the distribution of logits
        """
        probs = self.softmax(logits)
        dist = Categorical(probs)

        return dist.sample()
    
class TemperatureSampler(Sampler):
    """
    ## Sampler with Temperature
    """
    def __init__(self, temperature: float = 1.0):
        """
        :param temperature: is the temperature to sample with
        """
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """

        logits=logits / self.temperature
        probs = self.softmax(logits)
        # Create a categorical distribution with temperature adjusted logits
        dist = Categorical(probs)

        # Sample
        return dist.sample()
    
class TopKSampler(Sampler):
    """
    ## Top-k Sampler
    """
    def __init__(self, k: int, sampler: Sampler):
        """
        :param k: is the number of tokens to pick
        :param sampler: is the sampler to use for the top-k tokens

        `sampler` can be any sampler that takes a logits tensor as input and returns a token tensor;
         e.g. [`TemperatureSampler'](temperature.html).
        """
        self.k = k
        self.sampler = sampler

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """
        # New logits filled with $-\infty$; i.e. zero probability
        zeros = logits.new_ones(logits.shape) * float('-inf')
        # Pick the largest $k$ logits and their indices
        values, indices = torch.topk(logits, self.k, dim=-1)
        # Set the values of the top-k selected indices to actual logits.
        # Logits of other tokens remain $-\infty$
        zeros.scatter_(-1, indices, values)

        # Sample from the top-k logits with the specified sampler.
        return self.sampler(zeros)

class NucleusSampler(Sampler):
    """
    ## Nucleus Sampler
    """
    def __init__(self, p: float, sampler: Sampler):
        """
        :param p: is the sum of probabilities of tokens to pick $p$
        :param sampler: is the sampler to use for the selected tokens
        """
        self.p = p
        self.sampler = sampler
        # Softmax to compute $P(x_i | x_{1:i-1})$ from the logits
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits with Nucleus Sampling
        """

        # Get probabilities $P(x_i | x_{1:i-1})$
        probs = self.softmax(logits)

        # Sort probabilities in descending order
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        # Get the cumulative sum of probabilities in the sorted order
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Find the cumulative sums less than p.
        nucleus = cum_sum_probs < self.p
        # Prepend ones so that we add one token after the minimum number
        # of tokens with cumulative probability less that $p$.
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)

        # Get log probabilities and mask out the non-nucleus
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float('-inf')

        # Sample from the sampler
        sampled_sorted_indexes = self.sampler(sorted_log_probs)

        # Get the actual indexes
        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))

        #
        return res.squeeze(-1)