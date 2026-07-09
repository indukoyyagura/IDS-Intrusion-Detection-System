import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from models.student_teacher.teacher import Teacher
from models.student_teacher.student import Student

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

def main():
    input_dim = 20
    X = torch.randn(100, input_dim)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher = Teacher(input_dim).to(device)
    student = Student(input_dim).to(device)
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

    for epoch in range(5):
        student.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                teacher_logits = teacher(x_batch)

            student_logits = student(x_batch)
            loss = distillation_loss(student_logits, teacher_logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
