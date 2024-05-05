import torch
    
class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes a new Residual Block object
        
        h(x) = f(x) + x
        
        Args:
            in_channels (int): the number of input features of given layer
            out_channels (int): the number of output features of given layer
        """
        super(ResBlock, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = torch.nn.BatchNorm2d(out_channels)
        self.norm2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, 1, 2)
        
    def forward(self, x):
        
        
        residual = self.downsample(x)
       
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.silu(x)
        x = self.conv2(x)
        
        x = x + residual
        x = self.norm2(x)
        x = torch.nn.functional.silu(x)
        
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self):
        """
        Initializes a new Encoder object that encodes input image into latent space (compressed feature representation)
        """
        super(Encoder, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1)
        self.res1 = ResBlock(32, 64)
        self.res2 = ResBlock(64, 128)
        self.res3 = ResBlock(128, 256)
        self.res4 = ResBlock(256, 512)
        self.res5 = ResBlock(512, 512)

        self.meanpool1 = torch.nn.AvgPool2d(3, 2)
        self.meanpool2 = torch.nn.AvgPool2d(3, 2)
        
        self.norm1 = torch.nn.BatchNorm2d(32)
        
        #self.lin1 = torch.nn.Linear(516*4*4, 2048)
        #self.lin2 = torch.nn.Linear(2048, 512)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.silu(x)
        x = self.conv2(x)
        x = self.norm1(x)
        x = torch.nn.functional.silu(x)
        x = self.meanpool1(x)
        x = self.norm1(x)
        x = torch.nn.functional.silu(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.meanpool2(x)
        
        x = torch.flatten(x)
        
        #x = self.lin1(x)
        #x = torch.nn.functional.silu(x)
        #x = self.lin2(x)
        #x = torch.tanh(x)
        
        return x
 
class Deconv(torch.nn.Module):
    """
        Initializes a new Deconvolution (transposed convolution) object
        
        Args:
            in_channels (int): the number of input features of given layer
            out_channels (int): the number of output features of given layer
            kernel = size of the transposed convolution kernel (window)
            stride = size of the transposed convolution stride (step size)
            padding = specifise the usage of the padding (supplement of the image/feature map borders)
        """
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=0):
        super(Deconv, self).__init__()
        
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding)
        self.norm = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.norm(x)
        
        return torch.nn.functional.silu(x)
       
class Decoder(torch.nn.Module):
    def __init__(self):
        """
        Initializes a new Decoder object that decodes given latent space (compressed feature representation) into image
        """
        super(Decoder, self).__init__()
        
        self.lin1 = torch.nn.Linear(512, 2048)
        self.lin2 = torch.nn.Linear(2048, 516*4*4)
        
        self.dconv0 = Deconv(516, 512, 3, 2)
        self.dconv1 = Deconv(512, 256, 3, 2, 2)
        self.dconv2 = Deconv(256, 128, 3, 2, 2)
        self.dconv3 = Deconv(128, 128, 3, 2)
        self.dconv4 = Deconv(128, 64, 2, 2)
        self.dconv5 = Deconv(64, 64, 2, 2)
        self.dconv6 = Deconv(64, 32, 3, 1)
        self.dconv7 = torch.nn.ConvTranspose2d(32, 3, 3, 1)

        
    def forward(self, x):
        
        x = self.lin1(x)
        x = torch.nn.functional.silu(x)
        x = self.lin2(x)
        x = torch.nn.functional.silu(x)
        x = x.view(-1, 516, 4, 4)

        x = self.dconv0(x)
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        x = self.dconv5(x)
        x = self.dconv6(x)
        x = self.dconv7(x)
  
        x = torch.sigmoid(x)
        
        return x
    
class Autoencoder(torch.nn.Module):
    """
        Initializes a new Autoencoder object composed of Encoder and Decoder objects to compress and decompress an image
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        
        latent_space = self.encoder(x)
        x = self.decoder(latent_space)
        
        return x, latent_space
    
class VariationalAutoencoder(torch.nn.Module):
    def __init__(self):
        """
        Initializes a new Autoencoder object composed of Encoder and Decoder objects to compress image and decompress 
        from the sampling of latent space distribution
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.mu = torch.nn.Linear(512, 512)
        self.logvar = torch.nn.Linear(512, 512)
    
    def sample(self, mu, logvar):
        # given mean mu and log variance logvar, sample from distribution D(mu, logvar)

        logvar = torch.clamp(logvar, -20, 20)
        variance = torch.exp(logvar)
        std = variance.sqrt()

        return mu + std * torch.randn_like(std)
        
    def forward(self, x):
        
        latent_space = self.encoder(x)
        
        mu = self.mu(latent_space)
        logvar = self.logvar(latent_space)
        z = self.sample(mu, logvar)
        z *= 0.18
        
        x = self.decoder(z)
        
        return x, latent_space, mu, logvar