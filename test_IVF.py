from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction,Contourletfusion
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
from thop import profile
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/CDDFuse_03-07-20-08.pth"
for dataset_name in ["M3FD"]:
    print("\n"*2+"="*80)
    model_name="CDDFuse"
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder=os.path.join('test_result',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder().to(device))
    Decoder = nn.DataParallel(Restormer_Decoder().to(device))
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    # ContourletCNNLayer = nn.DataParallel(Contourletfusion(dim=64, num_heads=8)).to(device)
    
    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    # ContourletCNNLayer.load_state_dict(torch.load(ckpt_path)['ContourletCNNLayer'])
    
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    # ContourletCNNLayer.eval()
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)

    
    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir/images")):
            #start.record()
            data_IR=image_read_cv2(os.path.join(test_folder,"ir/images",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi/images",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
            
            flops, params = profile(Encoder, inputs=(data_VIS))
            # 打印FLOPs和参数量
            print('FLOPs = ' + str(flops / 1000**3) + 'G')
            print('Params = ' + str(params / 1000**2) + 'M')
            exit(0)
            
            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)

            # 裁剪 feature_I_D 的第3个维度到 450
            #if feature_I_D.shape[2] > feature_V_D.shape[2]:
                #feature_I_D = feature_I_D[:, :, :450, :]
            #if feature_I_B.shape[2] > feature_V_B.shape[2]:
                #feature_I_B = feature_I_B[:, :, :450, :]

            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            # feature_F_C = ContourletCNNLayer(feature_V_C + feature_I_C)
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8) #-----
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)
            #end.record()
            #torch.cuda.synchronize()
            #elapsed_time = start.elapsed_time(end)
            #print(f"Elapsed time: {elapsed_time} ms")





#     eval_folder=test_out_folder  
#     ori_img_folder=test_folder

#     metric_result = np.zeros((8))
#     for img_name in os.listdir(os.path.join(ori_img_folder,"ir/images")):
#             ir = image_read_cv2(os.path.join(ori_img_folder,"ir/images", img_name), 'GRAY')
#             ir = ir.astype(np.uint8)
#             vi = image_read_cv2(os.path.join(ori_img_folder,"vi/images", img_name), 'GRAY')
#             vi = vi.astype(np.uint8)
#             fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
#             fi = fi.astype(np.uint8)
#             metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
#                                         , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
#                                         , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
#                                         , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

#     metric_result /= len(os.listdir(eval_folder))
#     print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
#     print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
#             +str(np.round(metric_result[1], 2))+'\t'
#             +str(np.round(metric_result[2], 2))+'\t'
#             +str(np.round(metric_result[3], 2))+'\t'
#             +str(np.round(metric_result[4], 2))+'\t'
#             +str(np.round(metric_result[5], 2))+'\t'
#             +str(np.round(metric_result[6], 2))+'\t'
#             +str(np.round(metric_result[7], 2))
#             )
#     print("="*80)