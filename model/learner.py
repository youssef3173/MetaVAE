import torch
from torch import nn
from torch.nn import functional as F




# architecture: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
#############################################################################################################
# Encoder:
#############################################################################################################
class Encoder(nn.Module):
    # input size: B*3*128*128
    def __init__(self, config= None, in_channels=3, latent_dim= 128):
        super( Encoder, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        if config is None:
            self.config = []
        else:
            self.config = config


        in_channels = self.in_channels
        for h_dim in self.hidden_dims:
            self.config.append(
                ('conv2d', [h_dim, in_channels, 3, 3, 2, 1]) )
            self.config.append( 
                ('bn', [h_dim]))
            self.config.append( 
                ('leakyrelu', [0.01, False]))
            in_channels = h_dim

        self.config.append( [ 'flatten', [None]])
        self.config.append( [ 'final_layer', [self.latent_dim, self.hidden_dims[-1]*4*4]])

        # # ! if you use the 2 following lines , you need to change forward's return tensor
        # self.config.append( [ 'linear', [2*self.latent_dim, self.hidden_dims[-1]*4]])
        # self.config.append( [ 'reshape', [2, self.latent_dim]])

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.vars_bn_res = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'conv2d_res':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w0 = nn.Parameter(torch.ones(*param[0][:4]))
                torch.nn.init.kaiming_normal_(w0)
                self.vars.append(w0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                wb0 = nn.Parameter(torch.ones(param[0][0]))
                self.vars.append(wb0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                running_mean0 = nn.Parameter(torch.zeros(param[0][0]), requires_grad=False)
                running_var0 = nn.Parameter(torch.ones(param[0][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean0, running_var0])
                
                w1 = nn.Parameter(torch.ones(*param[1][:4]))
                torch.nn.init.kaiming_normal_(w1)
                self.vars.append(w1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))
                
                wb1 = nn.Parameter(torch.ones(param[1][0]))
                self.vars.append(wb1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))

                # must set requires_grad=False
                running_mean1 = nn.Parameter(torch.zeros(param[1][0]), requires_grad=False)
                running_var1 = nn.Parameter(torch.ones(param[1][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean1, running_var1])


            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name == 'final_layer':
                # mu:
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                # var:
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'in']:
                continue

            else:
                raise NotImplementedError

        self.inf0 = self.extra_repr()
        # print( '#----------------- Generator info: -----------------# \n', self.inf0 )



    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d, output_padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5], param[6],)
                info += tmp + '\n'
               
            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'

            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'

            elif name == 'conv2d_res':
                tmp = 'conv2d_res_1:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d) \t'\
                      %(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5],)
                tmp += 'conv2d_res_2:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5],)
                info += tmp + '\n'

            elif name == 'final_layer':
                tmp = 'final_layer->mu->linear:(ch_in:%d, ch_out:%d)'%(param[1], param[0])
                tmp = 'final_layer->var->linear:(ch_in:%d, ch_out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            else:
                raise NotImplementedError

        return info


    def Resforward(self, x, idx, bn_res_idx, param, vars, bn_training=True):
        w0, b0 = vars[idx], vars[idx + 1]
        # remember to keep synchrozied of forward_encoder and forward_decoder!
        x = F.conv2d(x, w0, b0, stride=param[0][4], padding=param[0][5])
        
        wb0, bb0 = vars[idx+2], vars[idx + 3]
        running_mean0, running_var0 = self.vars_bn_res[bn_res_idx], self.vars_bn_res[bn_res_idx+1]
        x = F.batch_norm(x, running_mean0, running_var0, weight=wb0, bias=bb0, training=bn_training)
        x = F.relu(x, inplace=param[0][0])
        
        w1, b1 = vars[idx + 4], vars[idx + 5]
        x = F.conv2d(x, w1, b1, stride=param[1][4], padding=param[1][5])
        wb1, bb1 = vars[idx+6], vars[idx + 7]
        running_mean1, running_var1 = self.vars_bn_res[bn_res_idx+2], self.vars_bn_res[bn_res_idx+3]
        x = F.batch_norm(x, running_mean1, running_var1, weight=wb1, bias=bb1, training=bn_training)
        return x


    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        bn_res_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)

            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5], output_padding = param[6])
                idx += 2
                # print(name, param, '\tout:', x.shape)

            elif name == 'conv2d_res':
                x = x + self.Resforward( x, idx, bn_res_idx, param, vars, True) 
                idx += 8
                bn_res_idx += 4

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())

            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                x = x.view(x.size(0), -1)

            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)

            elif name == 'relu':
                x = F.relu(x, inplace=param[0])

            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])

            elif name == 'tanh':
                x = torch.tanh(x)

            elif name == 'sigmoid':
                x = torch.sigmoid(x)

            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])

            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])

            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            elif name == 'final_layer':
                w, b = vars[idx], vars[idx + 1]
                mu = F.linear(x, w, b)
                idx += 2
                w, b = vars[idx], vars[idx + 1]
                var = F.linear(x, w, b)
                idx += 2

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        assert bn_res_idx == len(self.vars_bn_res)

        # return x
        return mu, var


    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()


    def parameters(self):
        return self.vars






#############################################################################################################
# Decoder:
#############################################################################################################
class Decoder(nn.Module):
    # input size: B*latent_dim
    def __init__(self, config = None, out_channels=3, latent_dim= 128):

        # To Do: Update encoder architecture and test it!
        super(Decoder, self).__init__()

        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dims = [32, 64, 128, 256, 512]

        # Build Decoder
        if config is None:
            self.config = []
        else:
            self.config = config

        # input layer:
        self.config.append( ('linear', [self.hidden_dims[-1]*4*4, self.latent_dim]) )
        self.config.append( ('reshape', [ self.hidden_dims[-1], 4, 4]) )

        # hidden layers:
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims)-1):
            self.config.append(
                ('convt2d', [self.hidden_dims[i], self.hidden_dims[i + 1], 3, 3, 2, 1, 1]) )
            self.config.append( 
                ('bn', [self.hidden_dims[i + 1]]))
            self.config.append( 
                ('leakyrelu', [0.01, False]))

        # final layer:
        self.config.append(
            ('convt2d', [self.hidden_dims[-1], self.hidden_dims[-1], 3, 3, 2, 1, 1]) )
        self.config.append(  
            ('bn', [self.hidden_dims[-1]]))
        self.config.append(  
            ('leakyrelu', [0.01, False]))
        self.config.append( 
            ('conv2d', [ self.out_channels, self.hidden_dims[-1], 3, 3, 1, 1]))
        self.config.append( 
            ('tanh', [True]))

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.vars_bn_res = nn.ParameterList()



        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'conv2d_res':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w0 = nn.Parameter(torch.ones(*param[0][:4]))
                torch.nn.init.kaiming_normal_(w0)
                self.vars.append(w0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                wb0 = nn.Parameter(torch.ones(param[0][0]))
                self.vars.append(wb0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                running_mean0 = nn.Parameter(torch.zeros(param[0][0]), requires_grad=False)
                running_var0 = nn.Parameter(torch.ones(param[0][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean0, running_var0])
                
                w1 = nn.Parameter(torch.ones(*param[1][:4]))
                torch.nn.init.kaiming_normal_(w1)
                self.vars.append(w1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))
                
                
                wb1 = nn.Parameter(torch.ones(param[1][0]))
                self.vars.append(wb1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))

                # must set requires_grad=False
                running_mean1 = nn.Parameter(torch.zeros(param[1][0]), requires_grad=False)
                running_var1 = nn.Parameter(torch.ones(param[1][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean1, running_var1])

            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'in']:
                continue
            else:
                raise NotImplementedError

        self.inf0 = self.extra_repr()
        # print( '#----------------- Discriminator info: -----------------# \n', self.inf0 )




    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d, output_padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5], param[6],)
                info += tmp + '\n'
               
            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'

            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'

            elif name == 'conv2d_res':
                tmp = 'conv2d_res_1:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d) \t'\
                      %(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5],)
                tmp += 'conv2d_res_2:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5],)
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def Resforward(self, x, idx, bn_res_idx, param, vars, bn_training=True):
        w0, b0 = vars[idx], vars[idx + 1]
        # remember to keep synchrozied of forward_encoder and forward_decoder!
        x = F.conv2d(x, w0, b0, stride=param[0][4], padding=param[0][5])
        
        wb0, bb0 = vars[idx+2], vars[idx + 3]
        running_mean0, running_var0 = self.vars_bn_res[bn_res_idx], self.vars_bn_res[bn_res_idx+1]
        x = F.batch_norm(x, running_mean0, running_var0, weight=wb0, bias=bb0, training=bn_training)
        x = F.relu(x, inplace=param[0][0])
        
        w1, b1 = vars[idx + 4], vars[idx + 5]
        x = F.conv2d(x, w1, b1, stride=param[1][4], padding=param[1][5])
        wb1, bb1 = vars[idx+6], vars[idx + 7]
        running_mean1, running_var1 = self.vars_bn_res[bn_res_idx+2], self.vars_bn_res[bn_res_idx+3]
        x = F.batch_norm(x, running_mean1, running_var1, weight=wb1, bias=bb1, training=bn_training)
        return x


    def forward(self, x, vars=None, bn_training=True):

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        bn_res_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5], output_padding=param[6])
                idx += 2
                # print(name, param, '\tout:', x.shape)

            elif name == 'conv2d_res':
                x = x + self.Resforward( x, idx, bn_res_idx, param, vars, True) 
                idx += 8
                bn_res_idx += 4

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = torch.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        assert bn_res_idx == len(self.vars_bn_res)


        return x


    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars

