import torch
import torch.nn as nn
import torch.nn.functional as F

# 1
print("ЗАДАНИЕ 1: SimpleModel через nn.Sequential")

model_sequential = nn.Sequential(
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.Tanh(),
    nn.Linear(16, 4),
    nn.Softmax(dim=1)
)

# 2
print("\nЗАДАНИЕ 2: Полносвязная сеть через nn.Module")

class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


fc_model = FCModel()

# 3
print("\nЗАДАНИЕ 3: Свёрточная сеть")

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x


conv_model = ConvModel()

# 4
print("\nЗАДАНИЕ 4: Объединённая модель (CNN + FC)")

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x


# Создание и тест модели
# model = CombinedModel()
# print(model)
#
# # Тестирование
# input_tensor = torch.randn(2, 3, 19, 19)
# output = model(input_tensor)
# print(f"\nТест модели:")
# print(f"Вход:  {input_tensor.shape}")
# print(f"Выход: {output.shape}")
# print(f"Пример выхода: {output[0]}")
# print(f"Сумма: {output[0].sum().item():.6f}")