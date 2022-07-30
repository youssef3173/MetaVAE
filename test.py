from options import Options
from model.learner import Decoder 

import torch
from torchvision.utils import save_image

import os



def denorm( x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


if __name__ == '__main__':
    opt = Options()
    opt.build()
    opt.isTrain = False

    save_root = 'results/'
    if not os.path.isdir( save_root):
        os.mkdir( save_root)

    device = torch.device( "cpu") 
    path_to_chkpt = f'checkpoints/mt_vae_results/FineTuning_chkpt.tar' 

    netDec = Decoder( latent_dim= opt.latent_dim)

    # print('Loading model ...')
    # state_dict = torch.load( path_to_chkpt, map_location=str( 'cpu' ) )
    # netDec.load_state_dict(state_dict['Dec_state_dict'])
    # print('...Done loading model')
    netDec.to( device )
    netDec.eval()
 
    batch_size = 5
    q_z = torch.empty( batch_size, opt.latent_dim).normal_( mean=0,std=1)
    p_x = netDec( q_z)


    xpath = os.path.join(save_root, 'tmp_img.jpg') 
    save_image( denorm( p_x.data.cpu()), xpath) 

