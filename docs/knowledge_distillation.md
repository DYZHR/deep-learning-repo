

# 代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transformers


class TeacherModel(nn.Module):
    def __init__(self, num_class=10):
        super(TeacherModel, self).__init__()
        self.features = nn.Sequential(
            # [b, 3, h, w] -> [b, 64, h, w]
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 64, h, w] -> [b, 64, h, w]
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 64, h, w] -> [b, 64, h/2, w/2]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [b, 64, h/2, w/2] -> [b, 128, h/2, w/2]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 128, h/2, w/2] -> [b, 128, h/2, w/2]
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 128, h/4, w/4] -> [b, 128, h/4, w/4]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [b, 256, h/4, w/4] -> [b, 256, h/4, w/4]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 256, h/4, w/4] -> [b, 256, h/4, w/4]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 256, h/4, w/4] -> [b, 256, h/8, w/8]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # cifar-10: h=w=32
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            # 默认0.5
            nn.Dropout(),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class StudentModel(nn.Module):
    def __init__(self, num_class=10):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            # [b, 3, h, w] -> [b, 32, h, w]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 32, h, w] -> [b, 32, h, w]
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 32, h, w] -> [b, 32, h/2, w/2]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [b, 32, h/2, w/2] -> [b, 64, h/2, w/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 64, h/2, w/2] -> [b, 64, h/2, w/2]
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 64, h/2, w/2] -> [b, 64, h/4, w/4]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DistillationLoss(nn.Module):
    def __init__(self, teacher_model, temperature=4.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_outputs, inputs, true_labels):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        # soft targets loss
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        # hard targets loss
        hard_loss = self.ce_loss(student_outputs, true_labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def evaluate_model(model, test_loader, device, model_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'{model_name} Accuracy: {accuracy:.2f}%')


def train_model_normal(model, model_name, train_loader, test_loader, device, epochs=10):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
        print(f"{model_name} Epoch [{epoch + 1} / {epochs}], Loss: {running_loss / len(train_loader):.4f}")

        evaluate_model(model, test_loader, device, model_name)


def train_student_with_distillation(student_model, teacher_model, train_loader, test_loader, device, epochs=10):
    teacher_model.eval()
    student_model.train()
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-3)
    distillation_criterion = DistillationLoss(teacher_model)

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            loss = distillation_criterion(student_outputs, inputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss
        print(f"Student Model with KD - Epoch [{epoch + 1} / {epochs}], Loss: {running_loss / len(train_loader):.4f}")
        evaluate_model(student_model, test_loader, device, "Student with KD")


def main():
    transform_train = transformers.Compose([
        transformers.RandomCrop(32, padding=4),
        transformers.RandomHorizontalFlip(),
        transformers.ToTensor(),
        transformers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transformers.Compose([
        transformers.ToTensor(),
        transformers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    teacher_model = TeacherModel().to(DEVICE)
    student_model = StudentModel().to(DEVICE)
    student_model_with_KD = StudentModel().to(DEVICE)

    print("\nStart Training Teacher Model")
    train_model_normal(
        teacher_model,
        "Teacher Model",
        train_loader,
        test_loader,
        DEVICE
    )
    print("\nEnd Training Teacher Model")

    print("\nStart Training Normal Student Model")
    train_model_normal(
        student_model,
        "Normal Student Model",
        train_loader,
        test_loader,
        DEVICE
    )
    print("\nEnd Training Student Model")

    print("\nStart Training Student Model with Knowledge Distillation")
    train_student_with_distillation(
        student_model_with_KD,
        teacher_model,
        train_loader,
        test_loader,
        DEVICE
    )
    print("\nEnd Training Student Model with Knowledge Distillation")


if __name__ == "__main__":
    main()
```