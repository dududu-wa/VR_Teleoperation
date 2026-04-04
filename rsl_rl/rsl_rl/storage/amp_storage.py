import torch


class AMPReplayBuffer:
    """Ring buffer for storing flattened AMP observations."""

    def __init__(self, buffer_size, amp_obs_size, device):
        self.buffer = torch.zeros(buffer_size, amp_obs_size, device=device)
        self.buffer_size = buffer_size
        self.insert_idx = 0
        self.count = 0

    def insert(self, amp_obs_batch):
        n = amp_obs_batch.shape[0]
        if n == 0:
            return

        if n >= self.buffer_size:
            self.buffer[:] = amp_obs_batch[-self.buffer_size :]
            self.insert_idx = 0
            self.count = self.buffer_size
            return

        end = self.insert_idx + n
        if end > self.buffer_size:
            overflow = end - self.buffer_size
            self.buffer[self.insert_idx :] = amp_obs_batch[: n - overflow]
            self.buffer[:overflow] = amp_obs_batch[n - overflow :]
        else:
            self.buffer[self.insert_idx : end] = amp_obs_batch

        self.insert_idx = end % self.buffer_size
        self.count = min(self.count + n, self.buffer_size)

    def sample(self, batch_size):
        if self.count == 0:
            raise RuntimeError("AMPReplayBuffer is empty, cannot sample.")
        idx = torch.randint(0, self.count, (batch_size,), device=self.buffer.device)
        return self.buffer[idx]
