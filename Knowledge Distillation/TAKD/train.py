import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from data.dataloader import get_data_loaders
from models.plain_cnn import ConvNet

plane_cifar10_book = {
	'2': ['Conv16', 'MaxPool', 'Conv16', 'MaxPool', 'FC10'],
	'4': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'FC10'],
	'6': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC10'],
	'8': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool',
		  'Conv128', 'Conv128','MaxPool', 'FC64', 'FC10'],
	'10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
		   'Conv256', 'Conv256', 'Conv256', 'Conv256' , 'MaxPool', 'FC128' ,'FC10'],
}

def get_args():
	parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
	parser.add_argument('--epochs', default=150, type=int,  help='number of total epochs to run')
	parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--teacher', default='10', type=str, help='teacher student name')
	parser.add_argument('--student', default='2', type=str, help='teacher student name')
	parser.add_argument('--device', default="cpu", type=str, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--checkpoint', default="10", type=str)
	args = parser.parse_args()
	return args

def train(student, teacher=None, config={}):
    
    train_loader, test_loader = get_data_loaders()
    lr = config['lr']
    momentum = config['momentum']
    device = config['device']
    lambda_ = config['lambda']
    T = config['T']
    epochs = config['epochs']
    trial_id = config['trial_id']

    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                              lr_lambda=lambda epoch: 0.1 if epoch < epochs // 2 else (0.01 if epoch < epochs * 3 // 4 else 0.001),
                                              last_epoch=-1,
                                              verbose=False)
    for epoch in range(epochs):
        student.train()
        for batch_idx, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = student(img)
            loss_SL = criterion(output, label) 
            loss = loss_SL
            
            if teacher:
                soft_target = teacher(img)
                loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1),
                                    F.softmax(soft_target / T, dim=1))
                loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD
            
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(f"epoch: {epoch}/{epochs}\tloss: {loss}")
        val_acc = validate(model=student, data_loader=test_loader, device=device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), f'./saved/{trial_id}.pth')
        
def validate(model, data_loader=None, device='cpu'):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
      for image,label in data_loader:
        image = image.to(device)
        label= label.to(device)
        output = model(image)
        _,output_index = torch.max(output,1)
        total += label.size(0)
        correct += (output_index == label).sum().float()
      accuracy = 100*correct/total
      print("Accuracy of Test Data: {}%".format(accuracy))
      return accuracy


if __name__ == "__main__":
    args = get_args()
    student_id = args.student
    teacher_id = args.teacher
    check_point = args.checkpoint
    
    teacher = ConvNet(plane_cifar10_book[teacher_id])
    teacher.load_state_dict(torch.load(check_point))
    student = ConvNet(plane_cifar10_book[student_id])
    train_config = {
		'epochs': args.epochs,
		'lr': args.learning_rate,
		'momentum': args.momentum,
		'device': args.cuda,
		'trial_id': teacher_id + "-" + student_id,
		'T': 5,
		'lambda_': 0.2,
	}
    train(student=student, teacher=teacher, config=train_config)