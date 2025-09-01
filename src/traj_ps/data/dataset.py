from torch.utils.data import Dataset

class DualTimelineDS(Dataset):
    def __init__(self, samples): self.samples = list(samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]
