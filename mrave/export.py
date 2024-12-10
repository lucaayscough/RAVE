import torch
from torch import nn
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, to_edge





class Snake(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha * x).sin().pow(2)






# Export and lower the module to Edge Dialect
example_args = (torch.ones(1),)
aten_dialect: ExportedProgram = export(Snake(2), example_args)
edge_program: EdgeProgramManager = to_edge(aten_dialect)
to_be_lowered_module = edge_program.exported_program()

from executorch.exir.backend.backend_api import LoweredBackendModule, to_backend

from backend import RaveBackend

lowered_module: LoweredBackendModule = to_backend(
    "RaveBackend", to_be_lowered_module, []
)

print(lowered_module)
print(lowered_module.backend_id)
print(lowered_module.processed_bytes)
print(lowered_module.original_module)

# Serialize and save it to a file
save_path = "delegate.pte"
with open(save_path, "wb") as f:
    f.write(lowered_module.buffer())
