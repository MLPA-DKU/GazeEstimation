from collections import defaultdict
import torch
import torch.optim as optim

__all__ = 'Lookahead'


class Lookahead(optim.Optimizer):

    def __init__(self, optimizer, k=5, alpha=0.5):

        if k < 0.0:
            raise ValueError('Invalid number of lookahead steps: {}'.format(k))
        if alpha < 0:
            raise ValueError('Invalid linear interpolation factor: {}'.format(alpha))

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group['counter'] = 0

    def _update(self, group):
        for fast in group['params']:
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.clone(fast.data).detach()

            slow = param_state['slow_param']
            fast.data.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
            slow.data.copy_(fast)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the models and returns the metrics.
        """
        loss = self.optimizer.step(closure=closure)

        for group in self.param_groups:
            if group['counter'] == 0:
                self._update(group)
            group['counter'] = (group['counter'] + 1) % self.k

        return loss

    def state_dict(self):
        r"""Returns the state of the optimizers as a :class:`dict`.
        It contains two entries:
        * state - a dict holding current optimization state. Its content
            differs between optimizers classes.
        * param_groups - a dict containing all parameter groups
        """
        slow_state_dict = super(Lookahead, self).state_dict()
        fast_state_dict = self.optimizer.state_dict()
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'fast_state': fast_state,
            'slow_state': slow_state_dict['state'],
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizers state.
        Arguments:
            state_dict: optimizers state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        fast_state_dict = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups'],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def zero_grad(self, set_to_none: bool = False):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        self.optimizer.zero_grad(set_to_none)

    def __repr__(self) -> str:
        base_str = self.optimizer.__repr__()
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += 'k: {}\n'.format(self.k)
        format_string += 'alpha: {}\n'.format(self.alpha)
        format_string += base_str
        format_string += '\n'
        format_string += ')'
        return format_string
