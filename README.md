# MAML VAE Implementation in Pytorch
MetaVAE for Few-shot Image Generation

# Dataset: Omniglot
The Omniglot data set is designed for developing more human-like learning algorithms. It contains 1623 different handwritten characters from 50 different alphabets. Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people.

# Usage
"""
$ python meta_train.py --name mt_vae_results --meta_dataroot omniglot-py/images_background/ --k_spt 5 --k_qry 5 --finetune_step 100 --num_epochs 500
"""

