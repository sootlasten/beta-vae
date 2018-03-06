from torch import nn
from torch.autograd import Variable


class DenseVAE(nn.Module):
    def __init__(self, nb_latents):
        super(DenseVAE, self).__init__()

        self.fc1 = nn.Linear(4096, 1200) 
        self.fc2 = nn.Linear(1200, 1200) 
        self.fc_mean = nn.Linear(1200, nb_latents)
        self.fc_std = nn.Linear(1200, nb_latents)
        self.fc3 = nn.Linear(nb_latents, 1200)
        self.fc4 = nn.Linear(1200, 1200)
        self.fc5 = nn.Linear(1200, 1200)
        self.fc6 = nn.Linear(1200, 4096)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc_mean(x), self.fc_std(x)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.tanh(self.fc3(z))
        x = self.tanh(self.fc4(x))
        # x = self.tanh(self.fc5(x))
        return self.sigmoid(self.fc6(x))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 64*64))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConvVAE(nn.Module):
    def __init__(self, nb_latents):
        super(ConvVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(64*4*4, 256)
        
        self.fc_mean = nn.Linear(256, nb_latents)
        self.fc_std = nn.Linear(256, nb_latents)
        
        self.fc2 = nn.Linear(nb_latents, 256)
        self.fc3 = nn.Linear(256, 64*4*4)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.fc1(x.view(-1, 64*4*4)))
        return self.fc_mean(x), self.fc_std(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        x = self.relu(self.fc2(z))
        x = self.relu(self.fc3(x))
        x = self.relu(self.deconv1(x.view(-1, 64, 4, 4)))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        return self.sigmoid(self.deconv4(x))
        
    def forward(self, x):
        mu, logvar = self.encode(x.unsqueeze(1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

