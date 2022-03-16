import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, in_channels =3 , num_classes = 1000, **kwargs ):
        super().__init__()
        self.conv1= nn.Sequential(
            nn.Conv2d(in_channels, out_channels =96, kernel_size= 11, bias = False, stride= 4 , padding =0),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ),
            nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, bias = False, padding = 2),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ),
            nn.BatchNorm2d(256),   #LRN 대신 BatchNorm 사용
            nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size= 3, bias = False, padding =2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size= 3, bias = False, padding =2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.FC = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        
    def forward(self, x):
        return self.FC(self.conv1(x).view(-1, 256 * 6 * 6))    
        
if __name__=="__main__":
    num_classes=1000
    IMAGE_SIZE=227
    model=AlexNet(num_classes=num_classes,)
    x = torch.randn((2,3,IMAGE_SIZE, IMAGE_SIZE))
    out=model(x)
    assert model(x)[0].shape == (2,3, 6, 6, num_classes+5)
    assert model(x)[1].shape == (2,3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes+5)
    assert model(x)[2].shape == (2,3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes+5)
    print("success!!!")