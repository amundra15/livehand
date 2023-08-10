import torch
import torch.nn as nn

from run_nerf_helpers import render
from models.superresolution import featuremap_to_rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImplicitRenderer(nn.Module):
    def __init__(self, **kwargs):
        super(ImplicitRenderer, self).__init__()
        
        self.mano_layer = kwargs['mano_layer']
        self.pyrenderer = kwargs['pyrenderer']
        
        ## helper parameters
        self.batch_size = kwargs['batch_size']
        self.chunk = kwargs['chunk']
        
        ## SR module parameters
        self.render_patches = kwargs['render_patches']
        self.render_full_image = kwargs['render_full_image']
        self.sr_module = kwargs['sr_module']
        self.H_render = kwargs['H_render']          #dimension at which the rays are shot
        self.W_render = kwargs['W_render']
        self.H = kwargs['H']                        #dimension of final image
        self.W = kwargs['W']
        self.sr_factor = kwargs['sr_factor']
        self.pose_cond_to_sr = kwargs['pose_cond_to_sr']
        
    
        
    def forward(self, batch_rays=None, c2w=None, bds=None, acc_mask=None, acc_mask_sr=None, mm_latent=None, mano_output=None,
                focal_render=None, t_gain=None, t_bias=None, **render_kwargs_train):

        #TODO: clean up this function

        if acc_mask is None:
            acc_mask = torch.ones(batch_rays.shape[-2], dtype=torch.bool, device=device)
        if acc_mask_sr is None:
            if self.render_full_image:
                acc_mask_sr = torch.ones(self.H, self.W, dtype=torch.bool, device=device)
            else:
                acc_mask_sr = torch.ones(self.H*self.W, dtype=torch.bool, device=device)


        output, disp, acc, extras = render(self.H_render, self.W_render, focal_render, chunk=self.chunk, rays=batch_rays, c2w=c2w,
                                                mm_latent=mm_latent, mano_output=mano_output,
                                                verbose=False, retraw=True, bds=bds,
                                                **render_kwargs_train)
        #print("output: ", output.shape)		#torch.Size([(N_rand, n_features])
        
        
        if self.render_full_image:
            #convert all tensors to 2D
            output = output.view(self.H_render, self.W_render, -1)
            acc = acc.view(self.H_render, self.W_render)
            acc_mask = acc_mask.view(self.H_render, self.W_render)
            

            if self.sr_factor > 1:
                sr_conditioning = mano_output['hand_pose'] if self.pose_cond_to_sr else None
                rgb, rgb_sr = featuremap_to_rgb(output, self.sr_module, device, conditioning=sr_conditioning, gt_H=self.H, gt_W=self.W)        
            
                rgb_sr[acc_mask_sr] = rgb_sr[acc_mask_sr] * t_gain + t_bias         #apply gain, bias only to the foreground
            
            else:
                rgb = output[...,:3]
            
        else:
            # rgb = output
            rgb = output[...,:3]
        
        if self.render_patches:
            rgb = rgb.view(64, 64, -1)
            acc = acc.view(64, 64, -1)
            acc_mask = acc_mask.view(64, 64)
            t_gain = t_gain.view(64, 64, -1)
            t_bias = t_bias.view(64, 64, -1)
        
        rgb[acc_mask] = rgb[acc_mask] * t_gain + t_bias         #apply gain, bias only to the foreground
        
        
        if 'rgb0' in extras:
            output0 = extras['rgb0']
            if self.render_full_image:
                output0 = output0.view(self.H_render, self.W_render, -1)
                acc0 = extras['acc0'].view(self.H_render, self.W_render)
                if self.sr_factor > 1:
                    rgb0, rgb0_sr = featuremap_to_rgb(output0, self.sr_module, device, conditioning=sr_conditioning, gt_H=self.H, gt_W=self.W)
                    rgb0_sr[acc_mask_sr] = rgb0_sr[acc_mask_sr] * t_gain + t_bias
                else:
                    rgb0 = output0[...,:3]
            else:
                # rgb0 = output0
                rgb0 = output0[...,:3]
            rgb0[acc_mask] = rgb0[acc_mask] * t_gain + t_bias         #apply gain, bias only to the foreground

        
        output = {'rgb': rgb, 'acc': acc}
        output.update(extras)
        
        if 'rgb_sr' in locals():
            output['rgb_sr'] = rgb_sr
        if 'rgb0' in locals():
            output['rgb0'] = rgb0
        if 'acc0' in locals():
            output['acc0'] = acc0
        if 'rgb0_sr' in locals():
            output['rgb0_sr'] = rgb0_sr
        
        return output