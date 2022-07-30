from torch.utils.data import  DataLoader


from options import Options
from model.model import MetaVAEModel
from data.meta_dataloader import meta_preprocess, MetaDataloader

import random


"""
Side Note:
if loss tends to 'nan' after some time,
 - Increase the number of batchs/tasks (opt.task_num)
 - Decrease Learnign Rate 
"""


if __name__ == '__main__':
    opt = Options()
    opt.build()

    root = opt.dataroot
    dictTrain, dataset_num = meta_preprocess(root)
    opt.test_indx_list = [random.randint( 0, dataset_num-1) for _ in range(opt.test_num_task)]

    dataset = MetaDataloader( k_shot=opt.k_spt, k_query=opt.k_qry,
                        resize=opt.load_size, dictTrain=dictTrain,
                        dataset_num=dataset_num, test_indx_list=opt.test_indx_list)   
    dataset_loader = DataLoader(dataset, opt.task_num, shuffle=True, num_workers=opt.num_threads, pin_memory=True)

    model = MetaVAEModel(opt)
    model.setup(opt) 
 

    # MT-VAE training:
    total_iters = len( dataset_loader )
    for epoch in range(2): #   ( opt.num_epochs):
        cur_iter = 0 
        for i, (state_spt, state_qry) in enumerate(dataset_loader): 
            model.set_input(state_spt, state_qry)        
            model.meta_train( epoch, cur_iter, total_iters)   
            cur_iter += 1
            
        # if epoch % 10 == 0 and epoch != 0:   
            suffix = 'Meta_chkpt.tar'
            model.save_networks( suffix, opt )
