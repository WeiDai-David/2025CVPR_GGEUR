# 用来确保模型的输出是10维
import torch
from torchvision.models import resnet50

def load_resnet_model():
    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)  # 确保全连接层为10个类别
    return model

def check_model(client_id):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_resnet_model().to(device)
    model.load_state_dict(torch.load(f'./CIFAR-10/features/resnet/initialmodel/resnet_cifar10_client_{client_id}.pth'))
    # model.load_state_dict(torch.load(f'./CIFAR-10/features/resnet/completemodel/resnet_cifar10_client_{client_id}.pth'))
    # 打印全连接层的形状
    fc_weight_shape = model.fc.weight.shape
    fc_bias_shape = model.fc.bias.shape
    print(f'fc.weight shape: {fc_weight_shape}')
    print(f'fc.bias shape: {fc_bias_shape}')

def main():
    # 检查所有客户端的模型
    for client_id in range(10):
        print(f'Checking model for client {client_id}')
        check_model(client_id)

if __name__ == "__main__":
    main()
