from torch.utils.data import  DataLoader

from options import Options
from model.model import MetaVAEModel
from data.meta_dataloader import meta_preprocess, FineTuningDataloader


if __name__ == '__main__':

    opt = Options()
    opt.build()
    opt.isTrain = False


    ftmodel = MetaVAEModel(opt)
    ftmodel.setup(opt) 

    suffix = 'Meta_chkpt.tar' 
    ftmodel.load_networks( suffix, opt )


    # root = opt.dataroot 
    root = opt.dataroot
    dictTrain, dataset_num = meta_preprocess(root)

    dataset_test = FineTuningDataloader( k_shot=opt.k_spt, k_query=opt.k_qry,
                        resize=opt.load_size, dictTrain=dictTrain,
                        dataset_num=dataset_num, test_dataset_indx= opt.test_indx_list[0])  
    dataset_loader_test = DataLoader(dataset_test, opt.task_num_val, shuffle=True, num_workers=opt.num_threads, pin_memory=True)



    # Fine Tuning:
    total_iters = len( dataset_loader_test )
    for epoch in range( 1):  # ( opt.test_epochs ):
        cur_iter = 0 
        for j, (state_spt, state_qry) in enumerate(dataset_loader_test): 
            ftmodel.set_input(state_spt, state_qry)
            ftmodel.finetunning( epoch, cur_iter, total_iters ) 
            cur_iter += 1

        suffix = 'FineTuning_chkpt.tar'
        ftmodel.save_networks( suffix, opt ) 
