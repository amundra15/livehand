import os

import torch
import torch.nn as nn

import lpips
from pytorch_fid import fid_score


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


class CustomLoss(nn.Module):
    def __init__(self, args, device):
        super(CustomLoss, self).__init__()
        
        self.args = args
        self.train_fine_network = args.N_importance > 0
        self.train_sr_module = args.sr_factor > 1
        self.use_perceptual_loss = args.perceptual_loss
        self.use_acc_loss = args.acc_loss
        self.use_opacity_reg = args.opacity_reg
        
        if self.use_perceptual_loss:
            if args.test_mode:
                self.lpips_loss = lpips.LPIPS(net='alex').eval().to(device)
            else:
                self.lpips_loss = lpips.LPIPS(net='vgg').to(device)


    def compute_all_losses(self, rgb, target, acc, acc_mask, rgb_sr, target_sr, output, network_label='fine_network'):
        
        loss_log = {}
        
        #rgb loss
        img_loss = img2mse(rgb, target)
        loss_log[f'{network_label}/img_loss'] = img_loss.item()
        psnr = mse2psnr(img_loss)
        loss_log[f'psnr/{network_label}'] = psnr.item()
        
        #perceptual loss
        if self.use_perceptual_loss:
            perceptual_loss = self.lpips_loss(rgb.permute(2,0,1).unsqueeze(0), target.permute(2,0,1).unsqueeze(0)).squeeze() * self.args.perceptual_loss_weight
            loss_log[f'{network_label}/perceptual_loss'] = perceptual_loss.item()
        
        #SR image loss
        if self.train_sr_module:
            img_loss_sr = img2mse(rgb_sr, target_sr)
            loss_log[f'{network_label}/img_loss_sr'] = img_loss_sr.item()
            img_loss += img_loss_sr
            psnr_sr = mse2psnr(img_loss_sr)
            loss_log[f'psnr/{network_label}_sr'] = psnr_sr.item()
            
            if self.use_perceptual_loss:
                perceptual_loss_sr = self.lpips_loss(rgb_sr.permute(2,0,1).unsqueeze(0), target_sr.permute(2,0,1).unsqueeze(0)).squeeze()
                perceptual_loss += self.args.perceptual_loss_weight * perceptual_loss_sr
                loss_log[f'{network_label}/perceptual_loss_sr'] = perceptual_loss_sr.item()
            
        if self.use_acc_loss:
            #background opacity loss
            acc_loss = (img2mse(acc[~acc_mask], 0) + img2mse(acc[acc_mask], 1)) * self.args.acc_loss_weight
            loss_log[f'{network_label}/bg_loss'] = acc_loss.item()

        if self.use_opacity_reg:
            #penalize the weights to be close to 0 or 1
            weights = output['weights'] if network_label == 'fine_network' else output['weights0']
            hard_loss = 2.5 * (-torch.log(torch.exp(-torch.abs(weights)) + torch.exp(-torch.abs(1-weights))).mean())
            loss_log[f'{network_label}/hard_loss'] = hard_loss.item()
        
        
        loss = img_loss 
        if self.use_perceptual_loss:
            loss += perceptual_loss 
        if self.use_acc_loss:
            loss += acc_loss
        if self.use_opacity_reg:
            loss += 2.5*hard_loss

        return loss, loss_log
    

    def forward(self, output, target):

        #get the output and ground truth
        rgb = output['rgb']
        target_s = target['rgb']
        acc = output['acc']
        acc_mask = target['acc']
        rgb_sr = output['rgb_sr'] if self.train_sr_module else None
        target_s_sr = target['rgb_sr'] if self.train_sr_module else None
        
        
        ## compute losses on the main network
        loss, loss_log = self.compute_all_losses(rgb, target_s, acc, acc_mask, rgb_sr, target_s_sr, output, 'fine_network')


        ## compute losses on the coarse network (if it exists)
        if self.train_fine_network:
            rgb0 = output['rgb0']
            acc0 = output['acc0']
            rgb0_sr = output['rgb0_sr'] if self.train_sr_module else None
            
            loss0, loss_log0 = self.compute_all_losses(rgb0, target_s, acc0, acc_mask, rgb0_sr, target_s_sr, output, 'coarse_network')

            loss += loss0
            loss_log.update(loss_log0)


        loss_log['loss'] = loss

        
        #debugging tool - plots the back-propogation graph
        # from torchviz import make_dot
        # make_dot(loss, params=dict(render_kwargs_train['network_fn'].named_parameters())).render("model_torchviz", format="png")
        # print("Gradient back-propogation tree saved.")

        return loss, loss_log



def calculate_fid_scores(gt_path, recon_path):
## for each subfolder in path1, find the corresponding subfolder in path2 and calculate the fid score
    
    fid_list = []
    
    for subfolder in os.listdir(recon_path):
        
        path1_sub = os.path.join(gt_path, subfolder, 'images')
        path2_sub = os.path.join(recon_path, subfolder)
        
        if os.path.exists(path1_sub) and os.path.exists(path2_sub):
            fid = fid_score.calculate_fid_given_paths([path1_sub, path2_sub], 128, 'cuda', 2048)
            fid_list.append(fid)
    
    return fid_list