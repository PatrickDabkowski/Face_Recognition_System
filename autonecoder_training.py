import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data import FaceDataset, Norm
from torchvision import transforms
from models import Autoencoder, VariationalAutoencoder

def training(X, model, epochs, device, criterion, optimizer, scheduler, mode="AE", beta=0.7, contrastive=False):
    """
        Train the model
        
        Args:
            X (data/tensor): training data points (tensor images)
            model (torch.nn.Module): PyTorch model
            epochs (int): number of training iterations at full training dataset
            device (torch.device): computational core ("CPU", "CUDA", "MPS")
            criterion: function measuring difference between model output and expected target value
            optimizer (torch.optim): function optimizing weights of model
            scheduler (torch.optim.lr_scheduler): function managing learning rate change during training 
            mode (str ["AE", "VAE"]): parameter that decides in what manner model should be trained
            beta (int): weight of the reconstruction loss (for KL divergence loss (1-beta))
        """
    # perform epoch
    previous_embedding = False
    negative_anchor = 0
    positive_anchor = 0
    
    for epoch in range(epochs):
        
        for i, x in enumerate(X):
            
            preson_id = x.pop(-1)
            img = torch.as_tensor(np.array(x)).view(5, 3, 224, 224).to(device)
            
            if mode == "AE":
                
                out, latent_space = model(img)
                loss = criterion(out, img)
            
            elif mode == "VAE":
                
                out, latent_space, mu, logvar = model(img)
                # Kullback-Leiber divergence for two Gaussian distributions
                KL =  torch.sum(torch.mean(-0.5*(1 + logvar - mu.pow(2) - logvar.exp())))
                # weighted loss 
                loss = beta*criterion(out, img) + (1-beta)*KL
            
            if contrastive:
                # first latent space is a reference
                # we minimise difference between latent spaces from same person
                positive_anchor = 0
                for j in range(1, latent_space.size(0)):
                    positive_anchor += criterion(latent_space[j], latent_space[0])
                positive_anchor /= latent_space.size(0) - 1
                
                # we maximise difference between latent spaces from different people
                if previous_embedding:
                    negative_anchor = -criterion(latent_space, previous_embedding)
                    previous_embedding = latent_space # for next iteration current person becomes a reference
                    
                loss += positive_anchor/2 + negative_anchor/2
                
            print(loss)
            
            # zero value of the gradient, prevents accumulation of gradients from different iterations
            optimizer.zero_grad()
            # computes the gradient of current tensor
            loss.backward() 
            # performs a single optimization step, update weights
            optimizer.step()

            # save results
            if i % 250 == 0:
                # change learning rate
                scheduler.step()
                plt.imsave(f'{args.results}/ref_epoch_'+str(epoch)+'_'+ str(i)+".png", img.squeeze().permute([0, 2,3,1]).cpu().detach().numpy()[0])
                plt.imsave(f'{args.results}/result_epoch_'+str(epoch)+'_'+ str(i)+".png", out.squeeze().permute([0, 2,3,1]).cpu().detach().numpy()[0])
            
            del x
            del out
            # release all unoccupied cached memory currently held by the caching allocator
            torch.mps.empty_cache()
        
        torch.save(model.state_dict(), args.target_path)
        
def test(X, model, device, criterion, mode="AE"):
    for i, x in enumerate(X):
        #transform(skimage.io.imread(x)).unsqueeze(0).to(device)
        preson_id = x.pop(-1)
        img = torch.as_tensor(np.array(x)).view(5, 3, 224, 224).to(device)
        
        if mode == "AE":
            
            out, latent_space = model(img)
            loss = criterion(out, img)
            print(f"Error: {error}")
        
        elif mode == "VAE":
            
            out, latent_space, mu, logvar = model(img)
            
            # Kullback-Leiber divergence for two Gaussian distributions
            KL =  torch.sum(torch.mean(-0.5*(1 + logvar - mu.pow(2) - logvar.exp())))
            # weighted loss 
            error = criterion(out, x) 

            print(f"Error: {error}\nKL divergence: {KL}")

        # save results

        plt.imsave('results/ref_epoch_'+str(test)+'_'+ str(i)+".png", x.squeeze().permute([1,2,0]).cpu().detach().numpy())
        plt.imsave('results/result_epoch_'+str(test)+'_'+ str(i)+".png", out.squeeze().permute([1,2,0]).cpu().detach().numpy())
        
        del x
        del out
        # release all unoccupied cached memory currently held by the caching allocator
        torch.mps.empty_cache()
            
if __name__ == "__main__":
    '''Used by me dataset is: https://microsoft.github.io/DigiFace1M/'''
    
    parser = argparse.ArgumentParser(description='Train AE/VAE')
    parser.add_argument('--dataloader_path', type=str, default='faces_dataloader.pt')
    parser.add_argument('--device', type=str, default='mps', help="computational unit to perform training ex. CUDA, MPS, CPU")
    parser.add_argument('--target_path', type=str, default='model.pt', help="destination path for saving trained AE/VAE")
    parser.add_argument('--model_type', type=str, default='AE', help="mode of model training AE/VAE")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs to train model")
    parser.add_argument('--results', type=str, default="results2", help="path to save visualisations")
    parser.add_argument('--contrastive', type=bool, default=True, help="usage of contrastive triplet loss during traning")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    criterion = torch.nn.MSELoss()
    epochs = args.epochs

    X = torch.load(args.dataloader_path)
    
    if args.model_type == "AE":
    
        model = Autoencoder().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        model.train()
        training(X, model, epochs, device, criterion, optimizer, scheduler, "AE", contrastive=args.contrastive)
        
    elif args.model_type == "VAE":
        
        model = VariationalAutoencoder().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        model.train()
        training(X, model, epochs, device, criterion, optimizer, scheduler, "AE", contrastive=args.contrastive)