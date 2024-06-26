import torch
    
class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initializes a new Residual Block object
        
        h(x) = f(x) + x
        
        Args:
            in_channels (int): the number of input features of given layer
            out_channels (int): the number of output features of given layer,
                for residual block it's recommended to change channel size together with activation map size (size of conv output)
        """
        super(ResBlock, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.downsample = None
        
        if stride > 1:
            self.downsample = torch.nn.Conv2d(in_channels, out_channels, 1, stride=2)
            
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(out_channels)
        self.norm2 = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
       
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)
        
        x = x + residual
        x = self.norm2(x)
        x = torch.nn.functional.leaky_relu(x)
        
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self):
        """
        Initializes a new Encoder object that encodes input image into latent space (compressed feature representation)
        """
        super(Encoder, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 64, 7, 2)
        self.norm1 = torch.nn.BatchNorm2d(64)
        self.maxpool = torch.nn.MaxPool2d(3, 2)
        
        self.res1 = ResBlock(64, 64)
        self.res2 = ResBlock(64, 64)
        self.res3 = ResBlock(64, 128, 2)
        self.res4 = ResBlock(128, 128)
        self.res5 = ResBlock(128, 128)
        self.res6 = ResBlock(128, 256, 2)
        self.res7 = ResBlock(256, 256)
        self.res8 = ResBlock(256, 256)
        self.res9 = ResBlock(256, 512, 2)
        self.res10 = ResBlock(512, 512)
        self.res11 = ResBlock(512, 512)
        self.res12 = ResBlock(512, 1024, 2)
        self.res13 = ResBlock(1024, 1024, 1)
        
        self.avgpool = torch.nn.AvgPool2d(3, 2)
        
        #self.lin1 = torch.nn.Linear(516*4*4, 2048)
        #self.lin2 = torch.nn.Linear(2048, 512)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.leaky_relu(x)
        
        x = self.maxpool(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        
        x = self.avgpool(x)
        
        x = torch.flatten(x)
        
        return x
 
class Deconv(torch.nn.Module):
    """
    Initializes a new Deconvolution (transposed convolution) object
    O = (I - 1) x stride - 2 x padding + kernel size
    
    Args:
        in_channels (int): the number of input features of given layer
        out_channels (int): the number of output features of given layer,
            for residual block it's recommended to change channel size together with activation map size (size of conv output)
        kernel = size of the transposed convolution kernel (window)
        stride = size of the transposed convolution stride (step size)
        padding = specifise the usage of the padding (supplement of the image/feature map borders)
        activation = wether use a activation function or not (in last layer it's recommended to not)
    """
    def __init__(self, in_channels, out_channels, stride=1, activation=True):
        super(Deconv, self).__init__()
            
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=1)
        self.upsample = None
        self.activation = activation

        if stride > 1:
            self.upsample = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            
        self.deconv2 = torch.nn.ConvTranspose2d(out_channels, out_channels, 2, stride=stride, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(out_channels)
        self.norm2 = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        if self.upsample:
            residual = self.upsample(x)
        else:
            residual = x

       
        x = self.deconv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.deconv2(x)
        x = x + residual
        x = self.norm2(x)
        
        if self.activation:
            x = torch.nn.functional.leaky_relu(x)
        
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        """
        Initializes a new Decoder object that decodes given latent space (compressed feature representation) into image
        """
        self.lin = torch.nn.Linear(1024, 1024*7*7)
        
        self.dconv1 = Deconv(1024, 512, 2)
        self.dconv2 = Deconv(512, 512, 1)
        self.dconv3 = Deconv(512, 512, 1)
        self.dconv4 = Deconv(512, 256, 2)
        self.dconv5 = Deconv(256, 256, 1)
        self.dconv6 = Deconv(256, 256, 1)
        self.dconv7 = Deconv(256, 128, 2)
        self.dconv8 = Deconv(128, 128, 1)
        self.dconv9 = Deconv(128, 128, 1)
        self.dconv10 = Deconv(128, 64, 2)
        self.dconv11 = Deconv(64, 64, 1)
        self.dconv12 = Deconv(64, 64, 1)
        self.dconv13 = Deconv(64, 3, 2, activation=False)
        #self.dconv6 = Deconv(64, 3, 2, activation=False)
        
    def forward(self, x):
        
        x = self.lin(x)
        x = x.view(-1, 1024, 7, 7)

        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        x = self.dconv5(x)
        x = self.dconv6(x)
        x = self.dconv7(x)
        x = self.dconv8(x)
        x = self.dconv9(x)
        x = self.dconv10(x)
        x = self.dconv11(x)
        x = self.dconv12(x)
        x = self.dconv13(x)
 
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

        logvar = torch.clamp(logvar, -20, 20) + 0.0001 * torch.randn_like(logvar)
        variance = torch.exp(logvar)
        std = variance.sqrt()

        return mu + std * torch.randn_like(std)
        
    def forward(self, x):
        
        latent_space = self.encoder(x)
        print(latent_space.size)
        
        mu = self.mu(latent_space)
        logvar = self.logvar(latent_space)
        z = self.sample(mu, logvar)
        z *= 0.18
        
        x = self.decoder(z)
        
        return x, latent_space, mu, logvar