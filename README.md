# MAML VAE Implementation in Pytorch
MetaVAE for Few-shot Image Generation, trained on Omniglot dataset

# Omniglot Dataset
The Omniglot data set is designed for developing more human-like learning algorithms. It contains 1623 different handwritten characters from 50 different alphabets. Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people.

# Usage

1. train Meta_VAE on 5-shot:
```
$ python meta_train.py --name mt_vae_results --meta_dataroot omniglot-py/images_background/ --k_spt 5 --k_qry 5 --update_step 100 --finetune_step 100 --num_epochs 500 
```
2. test Meta_VAE on 5-shot:
```
$ python fine_tuning.py --name mt_vae_results --meta_dataroot omniglot-py/images_background/ --k_spt 5 --k_qry 5 --update_step 100 --finetune_step 100 --test_epochs 50
```


