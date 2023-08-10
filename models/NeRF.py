import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, embedded_input_ch=3, embedded_input_ch_views=3, local_nerf_output_ch=4, skips=[4], use_viewdirs=False, latent_emb_size=24, embed_fn=None, embeddirs_fn=None,
                hand_pose_conditioning=False):
        """ 
        """
        super(NeRF, self).__init__()
        
        self.D = D
        self.W = W
        self.skips = skips
        
        self.embed_fn = embed_fn
        self.embedded_input_ch = embedded_input_ch
        self.use_viewdirs = use_viewdirs
        self.embeddirs_fn = embeddirs_fn
        self.embedded_input_ch_views = embedded_input_ch_views
        
        self.latent_emb_size = latent_emb_size
        self.hand_pose_conditioning = hand_pose_conditioning
        self.local_nerf_output_ch = local_nerf_output_ch
        
        
        nerf_input_size = embedded_input_ch+latent_emb_size
        if hand_pose_conditioning:
            mano_handpose_size = 48
            nerf_input_size += mano_handpose_size

        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_size, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W+nerf_input_size, W) for i in range(D-1)])

        if use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            self.views_linears = nn.ModuleList([nn.Linear(embedded_input_ch_views + W, W//2)])
            self.rgb_linear = nn.Linear(W//2, local_nerf_output_ch-1)       #-1 cos we already have 1 channel for alpha
        else:
            self.output_linear = nn.Linear(W, local_nerf_output_ch)


    def forward(self, x, additional_inputs):
        
        input_pts, input_views, embeddings = torch.split(x, [3, 3 if self.use_viewdirs else 0, self.latent_emb_size], dim=-1)
        pts_embedded = self.embed_fn(input_pts)
        inputs = torch.cat([pts_embedded, embeddings], -1)
        
        if self.hand_pose_conditioning:
            full_pose = additional_inputs["mano_output"]["full_pose"]
            full_pose_expanded = full_pose.expand(inputs.shape[0], -1)
            inputs = torch.cat([inputs, full_pose_expanded], dim=-1)

        #pass through the MLP
        h = inputs
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([inputs, h], -1)
        

        if self.use_viewdirs:
            #incorporate view dependant effects
            
            #estimate alpha
            alpha = self.alpha_linear(h)
            
            #estimate rgb
            feature = self.feature_linear(h)
            views_embedded = self.embeddirs_fn(input_views)
            h = torch.cat([feature, views_embedded], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            
            #final rgb and sigma
            outputs = torch.cat([rgb, alpha], -1)
            
        else:
            #estimate rgb and sigma
            outputs = self.output_linear(h)
        
        
        return outputs  
