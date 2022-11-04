import sys
import time

sys.path.append("../")

from model.textcnn import TextCNNConfig, TextCNN
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = TextCNN(TextCNNConfig.from_config_file("../common/configs/textcnn.conf.toml")).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss().to(device)

x = [torch.randint(0, 21128, (640, 1280)).to(device), 0]
y = torch.randint(0, 2, (640,)).to(device)

model.train()
now = time.time()
y_hat = model(x)
print("precision time： ", time.time() - now)

loss = criterion(y_hat, y)
optimizer.zero_grad()
loss.backward()
print("backward time： ", time.time() - now)

optimizer.step()
print("optim time： ", time.time() - now)

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(x)
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))