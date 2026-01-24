from torch._inductor.decomposition import decomps_to_exclude
import torch
from torch.utils._debug_mode import DebugMode

def run_model(model, data, *, compile_with=None):
    if compile_with is not None:
        model = torch.compile(model, backend=compile_with)
    with DebugMode(record_output=True) as dm, DebugMode.log_tensor_hashes(
        hash_inputs=True,
    ):
        dm_out = model(*data)
    return dm, dm_out

class Toy(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x).mm(x.T)

inputs = (torch.randn(4, 4),)
dm_eager, _ = run_model(Toy(), inputs)
print("Eager mode:")
print(dm_eager.debug_string())

"""

"""