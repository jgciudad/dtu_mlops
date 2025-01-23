import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    for i in range(10):
        model(inputs)
        prof.step()
            
# Print the top 10 operations sorted by total CPU time
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Print the top 30 operations sorted by total CPU time, grouped by input shape
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

# Print the top 30 operations sorted by self CPU memory usage
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))

prof.export_chrome_trace("trace.json")

