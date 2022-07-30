import torch
from torch.nn import functional as F
import itertools
from torchvision.utils import save_image
import os

from .learner import Encoder, Decoder
from .util import get_scheduler
 

###########################################################################################################
# import wandb
# wandb.init(project="MetaVAE", entity="MAMLs")
# wandb.config = { "CONFIG": None, }
###########################################################################################################


class MetaVAEModel():
    '''
    Our model is based on VAE's network architecture
    '''
    #######################################################################################################
    #######################################################################################################
    def __init__(self, opt):
        """Initialize the Meta-VAE class.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
            
        self.optimizers = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.update_step = opt.update_step
        self.update_lr = opt.update_lr
        self.finetune_step = opt.finetune_step
        ##################################################################################################

        self.model_names = ['Enc', 'Dec']

        self.netEnc = Encoder(in_channels=self.opt.input_nc, latent_dim=self.opt.latent_dim).to(self.device)
        self.netDec = Decoder(out_channels=self.opt.output_nc, latent_dim=self.opt.latent_dim).to(self.device)
 
        self.optimizer_VAE = torch.optim.Adam( itertools.chain(self.netEnc.parameters(), self.netDec.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.lr = opt.lr
        self.beta1 = opt.beta1

        if self.isTrain:
            self.experiment_name = os.path.join(opt.name,'train')
            train_save_path = os.path.join('./checkpoints',self.experiment_name, 'images/')
            if not os.path.exists(train_save_path): 
                os.makedirs(train_save_path)
        else:
            self.experiment_name = os.path.join(opt.name,'test')
            test_save_path = os.path.join('./checkpoints',self.experiment_name, 'images/')
            if not os.path.exists(test_save_path): 
                os.makedirs(test_save_path)


    ######################################################################################################
    # Helper methodes:
    ######################################################################################################
    def set_input(self, state_support, state_query):
        self.state_support = state_support.to(self.device)
        self.state_query = state_query.to(self.device)


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward_VAE(self, state, vars_Enc=None, vars_Dec=None):
        mu, logvar = self.netEnc(state, vars=vars_Enc, bn_training=True)
        q_z = self.reparameterize( mu, logvar)
        p_x = self.netDec( q_z, vars=vars_Dec,  bn_training=True)
        return mu, logvar, q_z, p_x


    def loss_VAE( self, state, mu, log_var, p_x, kld_weight = 0.00025 ):
        recons_loss = F.mse_loss( p_x, state)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        # return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        return loss


    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)


    #######################################################################################################
    # Meta Training:
    #######################################################################################################
    def meta_train(self,  epoch, cur_iter, total_iters):
        """MT-VAE training process"""
        task_num, setsz, c_, h, w = self.state_support.size()
        querysz = self.state_query.size(1)

        loss_q = 0

        for i in range(task_num):
            # Initialization
            self.state = self.state_support[i]
            self.state_q = self.state_support[i]

            # Forward pass:
            self.mu, self.logvar, self.q_z, self.p_x = self.forward_VAE( self.state)
            self.loss = self.loss_VAE(  self.state, self.mu, self.logvar, self.p_x)

            grad_Dec = torch.autograd.grad( self.loss, self.netDec.parameters(), retain_graph=True) # needed for Grad_Enc
            fast_weights_Dec = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_Dec, self.netDec.parameters())))
            grad_Enc = torch.autograd.grad( self.loss, self.netEnc.parameters())
            fast_weights_Enc = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_Enc, self.netEnc.parameters())))

            # Meta-traning over update_steps
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~update_step-1
                self.mu, self.logvar, self.q_z, self.p_x = self.forward_VAE( self.state, fast_weights_Enc, fast_weights_Dec)
                self.loss = self.loss_VAE(  self.state, self.mu, self.logvar, self.p_x)

                grad_Dec = torch.autograd.grad( self.loss, self.netDec.parameters(), retain_graph=True) # needed for Grad_Dec
                fast_weights_Dec = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_Dec, self.netDec.parameters())))
                grad_Enc = torch.autograd.grad( self.loss, self.netEnc.parameters())
                fast_weights_Enc = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_Enc, self.netEnc.parameters())))

                loss = {}
                loss['S_Meta_loss'] = self.loss.item()
                # wandb.log( loss )

                if (k+1) % 5 == 0:
                    log = "#-- [S] INTRA TASK TRINING --#, epoch [{}], Iteration [{}]/[{}] --#\n".format( epoch, cur_iter, total_iters)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)  
                    xpath = os.path.join('./checkpoints',self.experiment_name, 'images/', 'Meta_S_{}.jpg'.format( epoch)) 
                    save_image(self.denorm( self.p_x.data.cpu()), xpath)  

                
            #Meta-testing on the query set    
            self.mu, self.logvar, self.q_z, self.p_x = self.forward_VAE( self.state_q, fast_weights_Enc, fast_weights_Dec)
            self.loss_q = self.loss_VAE(  self.state_q, self.mu, self.logvar, self.p_x)

            loss_q +=  self.loss_q

            loss = {}
            loss['Q_Meta_loss'] = self.loss_q.item()
            # wandb.log( loss )

            log = "#-- [Q] INTRA TASK TRINING --#, epoch [{}], Iteration [{}]/[{}] --#\n".format( epoch, cur_iter, total_iters)
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log) 

        # optimize meta parameters
        self.optimizer_VAE.zero_grad()  
        loss_q.backward()           
        self.optimizer_VAE.step()      

        
###########################################################################################################
# Fine Tuning:
###########################################################################################################
    def finetunning(self, epoch, cur_iter, total_iters):
        """MT-VAE inference """
        task_num, setsz, c_, h, w = self.state_support.size()
        querysz = self.state_query.size(1)


        for i in range(task_num):
            # Initialization
            self.state = self.state_support[i]
            self.state_q = self.state_support[i]
            
            for k in range(1, self.finetune_step):
                self.mu, self.logvar, self.q_z, self.p_x = self.forward_VAE( self.state)
                self.loss = self.loss_VAE(  self.state, self.mu, self.logvar, self.p_x)  
                
                # optimize G_A and G_B: 
                self.optimizer_VAE.zero_grad()

                self.loss.backward()
                self.optimizer_VAE.step() 
                

                if (k+1) % 5 == 0:
                    loss = {}
                    loss['FTDA_loss_'] = self.loss.item()
                    # wandb.log( loss )

                    log = "#-- During fine tuning, epoch [{}], Iteration [{}]/[{}]/[{}] --#\n".format( epoch, k, cur_iter, total_iters)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

        if epoch % 2 == 0 and epoch != 0:
            _, _, _, self.p_x = self.forward_VAE( self.state_q) 
            xpath = os.path.join('./checkpoints',self.experiment_name, 'images/', 'FT_Q_{}_{}_{}.jpg'.format( epoch, cur_iter, k)) 
            save_image(self.denorm( self.p_x.data.cpu()), xpath)   
                
            
            
    #####################################################################################################
    # Helper methods:
    #####################################################################################################
    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            print( '#-------------  Fine Tuning mode -------------#' )
            # load_suffix = 'Meta_chkpt_[%d].tar' % (opt.epoch)
            # self.load_networks(load_suffix)
        self.print_networks( opt.print_net)
        

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()



    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_networks(self, save_suffix, opt):
        print('Saving model ...')
        state_dict = {}
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                state_dict[name+'_state_dict'] = net.state_dict()
        state_dict['test_indx_list'] = opt.test_indx_list
        path_to_chkpt = os.path.join(self.save_dir, save_suffix)
        torch.save( state_dict, path_to_chkpt)
        print('...Done saving model')

    def load_networks(self, load_suffix, opt):
        path_to_chkpt = os.path.join(self.save_dir, load_suffix)
        if os.path.isfile( path_to_chkpt):
            print('Loading model ...')
            state_dict = torch.load( path_to_chkpt, map_location=str(self.device))
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    net.load_state_dict(state_dict[name+'_state_dict'])
            opt.test_indx_list = state_dict['test_indx_list']
            print('...Done loading model')
        else:
            print( f' ... Loading failed, the file {path_to_chkpt} does not exist ... ')
            

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    #######################################################################################################
    # Print the networks:
    #######################################################################################################

    def print_networks( self, print_net ):
        print('#------------- Networks initialized ----------------#')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if print_net:
                    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                    print( f'#=====  net{name} info::  \n', net.inf0 )
        print('#---------------------------------------------------#')

