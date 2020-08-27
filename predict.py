import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

classLabels = ["desert", "mountains", "sea", "sunset", "trees" ]

model = torch.load('model.pt', map_location=torch.device('cpu'))
model = model.eval()

def get_tensor(img):
    tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    return tfms(Image.open(img)).unsqueeze(0)

def predict(img, label_lst, model):
    tnsr = get_tensor(img)
    op = model(tnsr)
    op_b = torch.round(op)
    
    op_b_np = torch.Tensor.cpu(op_b).detach().numpy()
    
    preds = np.where(op_b_np == 1)[1]
    
    sigs_op = torch.Tensor.cpu(torch.round((op)*100)).detach().numpy()[0]
    
    o_p = np.argsort(torch.Tensor.cpu(op).detach().numpy())[0][::-1]
    
    # print("Argsort: {}".format(o_p))
    # print("Softmax: {}".format(sigs_op))
    
    # print(preds)
    
    label = []
    for i in preds:
        label.append(label_lst[i])
        
    arg_s = {}
    for i in o_p:
#         arg_s.append(label_lst[int(i)])
        arg_s[label_lst[int(i)]] = sigs_op[int(i)]
    
    exist = []
    for i in range(5):
        if list(arg_s.items())[i][1] > 100:
            exist.append(list(arg_s.items())[i][0])
        
    print("existing object: {}".format(exist))
        
    # return label, list(arg_s.items())[:10]

photoName = "sunset.jpg"

path = "photo/" + photoName
image = plt.imread(path)
plt.imshow(image)
plt.show()
predict(path, classLabels, model)