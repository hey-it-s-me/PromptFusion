# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction, ContourletCNN,Contourletfusion
from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
import clip_score
import clip
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
criteria_fusion = Fusionloss()
model_str = 'PromptFusion'

# . Set the hyper-parameters for training
num_epochs = 120  # total epoch
epoch_gap = 0 # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.  # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
model.to('cuda')
for para in model.parameters():
    para.requires_grad = False

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

class Prompts(nn.Module):
    def __init__(self,prompts,initials=None):
        super(Prompts,self).__init__()
        self.text_encoder = TextEncoder(model)
        self.tokenized_prompts= clip.tokenize(prompts).cuda()
        if isinstance(initials,list):
            text = clip.tokenize(initials).cuda()
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
           self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding(clip.tokenize(["".join(["X"]*len(prompts)),"".join(["X"]*len(prompts))]).cuda()).requires_grad_())).cuda()

    def forward(self,tensor_list,flag=1):
        
        text_features = self.text_encoder(self.embedding_prompt,self.tokenized_prompts)
        
        score=0
        for i in range(len(tensor_list)):
            image_features=tensor_list[i]
            
            image_nor=image_features.norm(dim=-1, keepdim=True)
            nor= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * (image_features/image_nor) @ (text_features/nor).T).softmax(dim=-1)
            probs = similarity
            prob = probs[0][0]
            score =score + prob
        score=score/len(tensor_list)
        return score

clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
img_resize = transforms.Resize((224,224))

def get_image_feature(tensor):
    tensor_list = []
    for i in range(tensor.shape[0]):
        image2=img_resize(tensor[i])
        image3 = torch.cat([image2,image2,image2],dim=0)
        image4 = image3.unsqueeze(0)
        image=clip_normalizer(image4)
        
        image_features = model.encode_image(image)
        tensor_list.append(image_features)
        
    return tensor_list


A_description = "This image's fusion is very harmonious, the texture is very clear, the target is very striking."
B_description = "This image's fusion is not well integrated, the texture is weak, and then the target is fuzzy, not prominent, and the features are relatively smooth and flat."
text_encoder = TextEncoder(model)
A_prompts = Prompts(A_description)
B_prompts = Prompts(B_description)

# Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')

# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")




'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()
    
        
        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        
        if epoch < epoch_gap:  # Phase I
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            
            
            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
        else:  # Phase II
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)

            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)
            image_feature = get_image_feature(data_Fuse)
            A_similarity = A_prompts(image_feature)
            B_similarity = B_prompts(image_feature)
            loss_prompt = B_similarity/A_similarity
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)

            loss_decomp = (cc_loss_D) ** 2  / (1.01 + cc_loss_B)
            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            
            
            loss = fusionloss + coeff_decomp * loss_decomp +loss_prompt
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    scheduler1.step()
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()
    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    
if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/CDDFuse_" + timestamp + '.pth'))
