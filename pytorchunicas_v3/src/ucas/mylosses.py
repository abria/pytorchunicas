import torch
from torch.nn import functional as F

# loss function for Variational AutoEncoder (VAE)
# Reconstruction + KL divergence losses summed over all elements and batch
def VAE_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def RMSE_loss(outputs, inputs):
    avg_RMSE = torch.zeros(1, dtype=torch.float)
    for i in range(len(inputs)):
        avg_RMSE += (torch.sum((outputs[i]-inputs[i]).pow(2)))
    return (avg_RMSE/len(outputs)).sqrt()

# def AUC_loss(outputs, labels):
#     AUC = torch.zeros(1, dtype=torch.float, device='cuda:0')
#     pos = outputs[labels == 1]
#     neg = outputs[labels == 0]
#     for p in pos:
#         for n in neg:
#             if p > n:
#                 AUC += 1
#             elif p == n:
#                 AUC += 0.5
#     return 1-AUC
