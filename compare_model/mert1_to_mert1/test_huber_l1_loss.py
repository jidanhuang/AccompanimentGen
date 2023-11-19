
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子
torch.manual_seed(42)

# 定义模型
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 定义余弦损失函数
loss_fct =torch.nn.HuberLoss(reduction='mean')
# loss_fct =torch.nn.SmoothL1Loss(reduction='mean')

# 定义训练数据参数
batch_size = 4
sequence_length = 20
emb_dim = 10

# 创建模型实例
input_dim = emb_dim
output_dim = emb_dim
model = LinearModel(input_dim, output_dim)

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练数据
num_samples = batch_size
data = torch.randn(num_samples, sequence_length, emb_dim)
# labels = torch.randn(num_samples, sequence_length, emb_dim)
labels = data.clone()  # 使用相同的数据作为标签

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    data_flat = data.view(num_samples * sequence_length, -1)
    labels_flat = labels.view(num_samples * sequence_length, -1)
    
    outputs = model(data_flat)
    loss = loss_fct(outputs, labels_flat)
    loss.backward()
    print(loss)
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
