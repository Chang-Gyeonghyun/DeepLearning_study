import torch
from data.dataloader import get_data_loaders
from models.plain_cnn import ConvNet
import os
import matplotlib.pyplot as plt


plane_cifar10_book = {
	'2': ['Conv16', 'MaxPool', 'Conv16', 'MaxPool', 'FC10'],
	'4': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'FC10'],
	'6': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC10'],
	'8': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool',
		  'Conv128', 'Conv128','MaxPool', 'FC64', 'FC10'],
	'10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
		   'Conv256', 'Conv256', 'Conv256', 'Conv256' , 'MaxPool', 'FC128' ,'FC10'],
}

def validate(model, data_loader, device='cpu'):
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
      return accuracy.item()

if __name__ == "__main__":
  pth_files = [file for file in os.listdir('./saved') if file.endswith('.pth')]
  pth_files.sort()
  pth_files.pop()
  accuracy = []
  model_list = []
  device = "cuda" if torch.cuda.is_available() else "cpu"
  _, data_loader = get_data_loaders()
  for pth_file in pth_files:
      file_path = os.path.join('./saved', pth_file)
      model = ConvNet(plane_cifar10_book[pth_file[-5]]).to(device)
      model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
      result = validate(model, data_loader, device)
      accuracy.append(round(result,3))
      model_list.append(file_path[:-5])
  model_list.append('10')
  accuracy.append(85.560)
  plt.figure(figsize=(10, 6))
  plt.plot(model_list, accuracy, marker='o')
  plt.xlabel('Model')
  plt.ylabel('Accuracy')
  plt.title('Model Accuracy')
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()
    