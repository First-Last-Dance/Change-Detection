import torch
import torch.nn as nn
import torch.nn.functional as F  # Added import for functional module
from torch.nn.functional import relu, sigmoid

class ChangeDetectionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(6, 64, kernel_size=3 , padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3 , padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3 , padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3 , padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3 , padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3 , padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3 , padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3 , padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3 , padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3 , padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3 , padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3 , padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3 , padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3 , padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3 , padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3 , padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3 , padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3 , padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x1, x2):
        
        # Concatenate input images along the channel dimension
        x = torch.cat([x1, x2], dim=1)

        # Encoder
        xe11 = torch.relu(self.e11(x))
        xe12 = torch.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)


        xe21 = torch.relu(self.e21(xp1))
        xe22 = torch.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)


        xe31 = torch.relu(self.e31(xp2))
        xe32 = torch.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)


        xe41 = torch.relu(self.e41(xp3))
        xe42 = torch.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)


        xe51 = torch.relu(self.e51(xp4))
        xe52 = torch.relu(self.e52(xe51))


        # Decoder
        xu1 = self.upconv1(xe52)
        # xe42_resized = F.interpolate(xe42, size=xu1.size()[2:], mode='bilinear', align_corners=True)
        xu11 = torch.cat([xe42, xu1], dim=1)
        xd11 = torch.relu(self.d11(xu11))
        xd12 = torch.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        # xe32_resized = F.interpolate(xe32, size=xu2.size()[2:], mode='bilinear', align_corners=True)
        xu22 = torch.cat([xe32, xu2], dim=1)
        xd21 = torch.relu(self.d21(xu22))
        xd22 = torch.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        # xe22_resized = F.interpolate(xe22, size=xu3.size()[2:], mode='bilinear', align_corners=True)
        xu33 = torch.cat([xe22, xu3], dim=1)
        xd31 = torch.relu(self.d31(xu33))
        xd32 = torch.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        # xe12_resized = F.interpolate(xe12, size=xu4.size()[2:], mode='bilinear', align_corners=True)
        xu44 = torch.cat([xe12, xu4], dim=1)
        xd41 = torch.relu(self.d41(xu44))
        xd42 = torch.relu(self.d42(xd41))

        # Output layer with sigmoid activation
        out = torch.sigmoid(self.outconv(xd42))

        # print("out",out)
        # xu33 = torch.cat([xe22, xu4], dim=1)
        # out_binary = out > 0.5
        out_binary = torch.where(out > 0.5, torch.tensor(1.0, requires_grad= True), torch.tensor(0.0, requires_grad= True))
        # print("out_binary",out_binary)
        return out_binary



###################################


model = ChangeDetectionUNet()
model = model.cuda()

#################################

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

##############################

# Training loop
num_epochs = 5

total_true = []
total_pred = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_intersection = 0
    total_union = 0
    
    for batch_idx, (data_A, data_B, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data_A = data_A.float().cuda()
        data_B = data_B.float().cuda()
        target = target.float().cuda()
        output = model(data_A, data_B)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

        # print("Batch index train:",batch_idx)


    # Validation
    model.eval()
    total_val_loss = 0
    total_val_intersection = 0
    total_val_union = 0
    
    with torch.no_grad():
        for batch_idx, (data_A, data_B, target) in enumerate(val_loader):
            # print("Batch index test:",batch_idx)
            data_A = data_A.float().cuda()
            data_B = data_B.float().cuda()
            target = target.float().cuda()
            output = model(data_A, data_B)
            val_loss = criterion(output, target)
            total_val_loss += val_loss.item()

            total_pred.append(output)
            total_true.append(target)



    print(f'Epoch {epoch+1}, Train Loss: {total_train_loss/len(train_loader)}'
          f'Validation Loss: {total_val_loss/len(val_loader)}')
