import torch
import os
import argparse


"""
getattr(class, 'key') is equivalent to class.key
setattr(class, 'key', v) is equivalent to class.key = v
delattr(class, 'key') is equivalent to del class.key
"""


######################################################################################################
######################################################################################################

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

######################################################################################################
######################################################################################################

class Options():
    def __init__(self):
        self.initialized = False

    ##################################################################################################
    ##################################################################################################
    def initialize( self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--preprocess", type= str, default= 'resize_and_crop')                          
        parser.add_argument("--beta1", type= float, default= 0.5)                           
        parser.add_argument("--checkpoints_dir", type= str, default= './checkpoints')                 
        parser.add_argument("--continue_train", action='store_true')                                                
        parser.add_argument("--epoch", type= str, default= 'latest')                        
        parser.add_argument("--epoch_count", type= int, default= 1)    
        parser.add_argument("--test_num_task", type= int, default= 5)                        
        parser.add_argument("--finetune_step", type= int, default= 100)    
        parser.add_argument("--update_step", type= int, default= 10)                 
        parser.add_argument("--gpu_ids", type= str, default= '1' )                                             
        parser.add_argument("--input_nc", type= int, default= 3) 
        parser.add_argument("--output_nc", type= int, default= 3)                            
        parser.add_argument("--isTrain", action='store_false')
        parser.add_argument("--k_qry", type= int, default= 5)             
        parser.add_argument("--k_spt", type= int, default= 5)                                                  
        parser.add_argument("--load_iter", type= int, default= 500)
        parser.add_argument("--load_size", type= int, default= 128)                        
        parser.add_argument("--lr", type= float, default= 0.0002)                                                  
        parser.add_argument("--lr_policy", type= str, default= 'linear')               
        parser.add_argument("--lr_decay_iters", type= int, default= 50 )
        parser.add_argument("--niter", type= int, default= 100 )
        parser.add_argument("--niter_decay", type= int, default= 100 )                              
        parser.add_argument("--dataroot", type= str, default= 'omniglot-py/images_background/')    

        parser.add_argument("--weight_decay", type= float, default= 0.0)
        parser.add_argument("--scheduler_gamma", type= float, default= 0.95)
        parser.add_argument("--kld_weight", type= float, default= 0.00025)
        parser.add_argument("--latent_dim", type= int, default= 128)

        parser.add_argument("--test_indx_list", type= list, default= [])

        parser.add_argument("--model", type= str, default= 'mt_vae')                              
        parser.add_argument("--name", type= str, default= 'mt_vae_results')                                                                   
        parser.add_argument("--num_threads", type= int, default= 4)                                                    
        parser.add_argument("--phase", type= str, default= 'train')                            
        parser.add_argument("--suffix", type= str, default= '')                             
        parser.add_argument("--task_num", type= int, default= 64)                 
        parser.add_argument("--task_num_val", type= int, default= 1)                 
        parser.add_argument("--update_lr", type= float, default= 0.0001)                              
        parser.add_argument("--print_net", action='store_false')
        parser.add_argument("--no_dropout", action='store_false')

        parser.add_argument("--num_epochs", type= int, default= 100)
        parser.add_argument("--test_epochs", type= int, default= 10)

        args = parser.parse_args()
        for attribute, value in args.__dict__.items():
            setattr( self, str(attribute), value )
        
        self.initialized = True
    
    ###################################################################################################
    ###################################################################################################
    def print_options( self ):
        message = ''
        message += '#-------------------- Options ------------------#\n'
        for attribute, value in self.__dict__.items():
            message += '{:>25}: {:<30}\n'.format( str(attribute), str(value) )
        message += '#-------------------- End ----------------------#'
        print(message)
        
        expr_dir = os.path.join( self.checkpoints_dir, self.name )
        mkdirs( expr_dir)
        file_name = os.path.join( expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write( message )
            opt_file.write('\n')

            
    def build( self ):
        # Initialize:
        self.initialize()
        # GPUs set_up:
        str_ids = self.gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            if str_id != '':
                id = int(str_id)
                if id >= 0:
                    self.gpu_ids.append(id)
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])
        # Print Options:
        self.print_options()





"""
  -> Convert a dict to class:
"""
"""
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


dict = {'A': 13, 'B':17, 'C': 53}
bC = Dict2Class( dict )

print( bC.A )
print( bC.B )
print( bC.C )
"""