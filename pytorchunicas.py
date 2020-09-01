import sys
import multiprocessing
import argparse
import os
import inspect
import math
import time
import datetime
import inspect
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
#from torchvision import datasets, transforms
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from ucas import mynets, mylosses, preprocessing, augmentation, CvROI

# define top-level parser and subparsers
PARSER_GLOBAL = argparse.ArgumentParser(description='PyTorch Unicas Experimenter Tool')
SUBPARSERS = PARSER_GLOBAL.add_subparsers(
    help='Choose one of these modalities:', dest='command', title="subcommands", metavar="<command>")
PARSER_TRAIN = SUBPARSERS.add_parser('train',
    help='Train or finetune a model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
PARSER_TEST = SUBPARSERS.add_parser('test',
    help='Score a model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
PARSER_CROSSVALID = SUBPARSERS.add_parser('crossvalid',
    help='Cross-validation (includes multiple iterations of training and testing)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# define common arguments
for subparser in [PARSER_TRAIN, PARSER_TEST, PARSER_CROSSVALID]:
    required_args = subparser.add_argument_group('required named arguments')
    required_args.add_argument('--model', type=str, choices=[m[0] for m in inspect.getmembers(mynets, inspect.isclass)], required=True,
        help='Model architecture.')
    required_args.add_argument('--roi_files', default=argparse.SUPPRESS, type=lambda s: [str(item) for item in s.split(',')], required=True,
        help='Roi files separated by \',\' one for each class.')
    required_args.add_argument('--img_folder', default=argparse.SUPPRESS, type=str, required=True,
        help='Directory with all the images.')
    required_args.add_argument('--workspace', default=argparse.SUPPRESS, type=str, required=True,
        help='Folder containing input and output files.')
    subparser.add_argument('-b', '--batch_size', default=32, type=int,
        help='mini-batch size')
    subparser.add_argument('--gpu', default=argparse.SUPPRESS, type=int,
        help='GPU id to use.')
    subparser.add_argument('--img_prefix', default='', type=str,
        help='Prefix to be added to all the images.')
    subparser.add_argument('--img_suffix', default='', type=str,
        help='Suffix to be added to all the images.')
    subparser.add_argument('--img_list', default=argparse.SUPPRESS, type=str,
        help='Image inclusion list (only load samples if they belong to the images contained in this list).')
    subparser.add_argument('--img_channel', default=argparse.SUPPRESS, type=int,
        help='Image channel selection according to BGR color space indexing (B = 0, G = 1, R = 2).')
    subparser.add_argument('--max_epochs', default=30, type=int,
        help='Maximum number of epochs to train or test. In the latter case, this determines the model to load.')

# train only arguments
for subparser in [PARSER_TRAIN, PARSER_CROSSVALID]:
    subparser.add_argument('--class_weights', default=argparse.SUPPRESS, type=lambda s: [float(item) for item in s.split(',')],
        help='Class weights determining the class sampling probability at training time.')
    subparser.add_argument('--class_max_counts', default=argparse.SUPPRESS, type=lambda s: [int(item) for item in s.split(',')],
        help='Max number of samples that can be loaded, for each class.')
    subparser.add_argument('--class_min_counts', default=argparse.SUPPRESS, type=lambda s: [int(item) for item in s.split(',')],
        help='Min number of samples that can be loaded, for each class.')
    subparser.add_argument('--preprocessing', default='PixelStandardization', type=str, choices=[m[0] for m in inspect.getmembers(preprocessing, inspect.isclass) if m[0] not in ['Preprocessing']], required=False,
        help='Data preprocessing.')
    subparser.add_argument('--augmentation', default='FlipRotate', type=str, choices=[m[0] for m in inspect.getmembers(augmentation, inspect.isclass)], required=False,
        help='Data augmentation.')
    subparser.add_argument('--optimizer', default='SGD', type=str, choices=[m[0] for m in inspect.getmembers(torch.optim, inspect.isclass) if m[0] not in ['Optimizer']], required=False,
        help='Optimizer.')
    subparser.add_argument('--lr_base', default=0.001, type=float,
        help='Base learning rate.')
    #subparser.add_argument('--lr_scheduler', default = 'StepLR', type=str, choices = [m[0] for m in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass) if m[0] not in ['_LRScheduler', 'partial', 'Optimizer']], required=False,
    #   help='Learning rate scheduler.')
    subparser.add_argument('--lr_stepsize', default=6, type=int,
        help='Period of learning rate decay (in epochs).')
    subparser.add_argument('--lr_gamma', default=0.1, type=float,
        help='Multiplicative factor of learning rate decay.')
    subparser.add_argument('--momentum', default=0.9, type=float,
        help='Momentum factor.')
    subparser.add_argument('--weight_decay', default=0, type=float,
        help='Weight decay (L2 penalty).')
    subparser.add_argument('--display', default=0.1, type=float,
        help='The number of epochs between displaying info.')
    subparser.add_argument('--loss_weights', default=argparse.SUPPRESS, type=lambda s: [float(item) for item in s.split(',')],
        help='Class weights assigned when computing the loss function (weights < 0 will be assigned automatically)')
    subparser.add_argument('--resume', action='store_true',
        help='Resume training from last epoch model.')
    subparser.add_argument('--start_epoch', default=0, type=int, metavar='N',
        help='Manual epoch number (useful on restarts)')
    subparser.add_argument('-j', '--num_workers', default=2, type=int,
        help='Number of workers for mini batch loading')
    subparser.add_argument('--no_reshuffle', action='store_true',
        help='Disables rehuffling of training samples at the beginning of each epoch.')
    subparser.add_argument('--data_stats', action='store_true',
        help='Print data statistics (min, max, mean, std) during training')

# test only arguments
# --- none so far!

# crossvalid only arguments
PARSER_CROSSVALID.add_argument('--cross_valid_N', default=2, type=int,
                        help='Number of cross-validation folds.')
PARSER_CROSSVALID.add_argument('--fold_start', default=1, type=int,
                        help='Fold start')
PARSER_CROSSVALID.add_argument('--fold_end', default=-1, type=int,
                        help='Fold end')
PARSER_CROSSVALID.add_argument('--skip_training', action='store_true',
                    help='Skip training phase.')
PARSER_CROSSVALID.add_argument('--skip_testing', action='store_true',
                    help='Skip testing phase.')

# utility function to load dataset
def load_dataset(is_train, is_crossvalid, preprocessing_fun, is_autoencoder=False):
    return CvROI(
        roi_files=[ARGS.roi_files[0]] if is_autoencoder else ARGS.roi_files,
        img_folder=ARGS.img_folder, 
        img_prefix=ARGS.img_prefix, 
        img_suffix=ARGS.img_suffix,
        img_channel=ARGS.img_channel if hasattr(ARGS, 'img_channel') else None,
        img_list=ARGS.img_list,
        train=is_train,
        crossvalid=is_crossvalid,
        class_weights=ARGS.class_weights if hasattr(ARGS, 'class_weights') else [],
        class_max_counts=ARGS.class_max_counts if hasattr(ARGS, 'class_max_counts') else [],
        class_min_counts=ARGS.class_min_counts if hasattr(ARGS, 'class_min_counts') else [],
        augmentation=getattr(augmentation, ARGS.augmentation)() if train and not is_autoencoder else None,
        preprocessing=preprocessing_fun)

# train function
def train(path, cross_validation_iter=None):
    
    print '\nTraining STARTED, experiment %s' % (os.path.basename(os.path.normpath(ARGS.workspace)),)

    t0_train = time.time()
    print '\nTraining parameters:'
    print '...device              = ' + str(DEVICE)
    print '...num workers         = ' + str(ARGS.num_workers)
    print '...reshuffle           = ' + str(not ARGS.no_reshuffle)

    # instantiate net
    net = getattr(mynets, ARGS.model)()
    print '...net                 = ' + ARGS.model
    conv_autoencoder = False
    var_autoencoder = False
    if "CAE" in ARGS.model:
        conv_autoencoder = True
    elif 'VAE' in ARGS.model:
        var_autoencoder = True
    print '...batch size          = ' + str(ARGS.batch_size)

    # instantiate loss
    if conv_autoencoder:
        criterion = nn.MSELoss(reduction='sum')
        print '...loss                = ' + 'MSE'
    elif var_autoencoder:
        print '...loss                = BCE + KLD for VAE'
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(ARGS.loss_weights)).float().cuda(device=DEVICE) if hasattr(ARGS, 'loss_weights') else None)
        print '...loss                = ' + 'CrossEntropy' + (repr(ARGS.loss_weights) if hasattr(ARGS, 'loss_weights') else '')
    
    # instantiate optimizer
    if ARGS.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=ARGS.lr_base, momentum=ARGS.momentum, weight_decay=ARGS.weight_decay)
        print '...solver              = ' + 'SGD'
        print '...momentum            = ' + str(ARGS.momentum)
        print '...weight_decay        = ' + str(ARGS.weight_decay)
    elif ARGS.optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=ARGS.lr_base, weight_decay=ARGS.weight_decay)
        print '...solver              = ' + 'Adam'
        print '...betas               = ' + '0.9, 0.999'
        print '...eps                 = ' + '1e-8'
        print '...amsgrad             = ' + 'False'
        print '...weight_decay        = ' + str(ARGS.weight_decay)
        ARGS.lr_gamma = 1
    else:
        raise NotImplementedError

    # instantiate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=ARGS.lr_stepsize, gamma=ARGS.lr_gamma)
    print '...lr_base             = ' + str(ARGS.lr_base)
    print '...lr_policy           = ' + 'StepLR'
    print '...lr_gamma            = ' + str(ARGS.lr_gamma)
    print '...lr_stepsize         = ' + str(ARGS.lr_stepsize)

    # optionally resume from a checkpoint
    start_epoch = ARGS.start_epoch
    if ARGS.resume:
        print '\nResume from last epoch file'
        models = sorted(glob.glob(path + '/_epoch*.tar'))
        if not models:
            print '...no files matching _epoch*.tar pattern were found'
        else:
            if len(models) >= ARGS.max_epochs:
                print '...last epoch file (%d) is ahead of max_epochs (%d)\n\nTraining NOT REQUIRED' % (len(models), ARGS.max_epochs)
                return

            print '...found %d checkpoints, resume from %s' %(len(models), os.path.basename(models[-1]))
            checkpoint = torch.load(models[-1], map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch']
            net = checkpoint['net']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    # instantiate preprocessing
    prepro = getattr(preprocessing, ARGS.preprocessing)()

    # load dataset
    dataset = load_dataset(True, (cross_validation_iter, ARGS.cross_valid_N), prepro, conv_autoencoder)
    nchans = dataset.n_channels
    width = dataset.roi_w
    height = dataset.roi_h
    # dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
    # nchans = 1
    # width = 28
    # height = 28

    # instantiate data loader
    dataloader = DataLoader(dataset, batch_size=ARGS.batch_size, shuffle=not ARGS.no_reshuffle, num_workers=ARGS.num_workers, pin_memory=True)

    # save preprocessing parameters
    metadata_folder = path + "/.metadata"
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder)
    prepro.save(metadata_folder)

    # create reconstruction folder for autoencoder mode
    autoencoder_folder = path + "/reconstruction"
    if conv_autoencoder or var_autoencoder:
        if not os.path.exists(autoencoder_folder):
            os.makedirs(autoencoder_folder)

    # move net and optimizer to cuda
    net.to(DEVICE)
    for state in optimizer.state.values():
        for k, ten in state.items():
            if torch.is_tensor(ten):
                state[k] = ten.cuda(device=DEVICE)

    # prepare stat file
    with open(path + '/.stats.txt', 'w') as stats_f:
        stats_f.write('Epoch Loss LR\n')

    # switch to train mode
    net.train()

    # train
    display_iters = round(len(dataloader)*ARGS.display)
    t0_epoch = time.time()
    losses = []
    losses_ticks = []
    loss_min = float('inf')
    epoch_loss_min = 0
    for epoch in range(start_epoch, ARGS.max_epochs):

        # step scheduler
        scheduler.step()

        # reset loss
        running_loss = 0.0
        
        # reset times
        t0_epoch_loop = time.time()
        total_time_forward = 0
        total_time_backward = 0
        total_time_opt = 0
        total_time_data_load = 0
        total_time_other = 0
        t0_data_load = time.time()

        # reset statistics
        inputs_sum = 0
        inputs_sum_sq = 0
        inputs_sum_norm_f = ARGS.batch_size * width * height * nchans
        inputs_min = float('+inf')
        inputs_max = float('-inf')

        # 1 epoch = 1 complete loop over the dataset
        for i, data in enumerate(dataloader, 0):

            total_time_data_load += time.time() - t0_data_load

            # get the inputs
            t0_other = time.time()
            inputs, labels = data
            #labels = labels[..., np.newaxis].float()
            
            # send to GPU
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # update data statistics
            if ARGS.data_stats:
                inputs_sum += inputs.sum().detach().cpu()
                inputs_sum_sq += inputs.pow(2).sum().detach().cpu()
                inputs_min = min(inputs_min, inputs.min().detach().cpu())
                inputs_max = max(inputs_max, inputs.max().detach().cpu())

            # zero the parameter gradients
            optimizer.zero_grad()
            total_time_other += time.time() - t0_other

            # forward
            t0_forward = time.time()
            if var_autoencoder:
                outputs, mu, logvar = net(inputs)
            else:
                outputs = net(inputs)
            total_time_forward += time.time() - t0_forward

            # backward
            t0_backward = time.time()
            if conv_autoencoder:
                loss = criterion(outputs, inputs)
            elif var_autoencoder:
                loss = mylosses.VAE_loss(outputs, inputs, mu, logvar)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            total_time_backward += time.time() - t0_backward

            # optimizer
            t0_opt = time.time()
            optimizer.step()
            total_time_opt += time.time() - t0_opt

            # accumulate loss over 'display_iters' iterations
            running_loss += loss.item()

            # print/save statistics every 'display_iters' iterations
            if i % display_iters == display_iters-1:

                # average loss along last 'display_iters' iterations
                loss_avg = running_loss / display_iters

                # print statistics
                data_stats_str = ''
                if ARGS.data_stats:
                    inputs_mean = inputs_sum/((i+1)*inputs_sum_norm_f)
                    inputs_mean_sq = inputs_sum_sq/((i+1)*inputs_sum_norm_f)
                    data_stats_str = ('\n...data in [%.1f, %.1f] and centered on %.1f +/- %.1f' % (inputs_min, inputs_max, inputs_mean, (inputs_mean_sq - inputs_mean**2)**0.5)) if ARGS.data_stats else ''
                autoencoder_str = ''
                if conv_autoencoder or var_autoencoder:
                    avg_RMSE = 0
                    for inp, out in zip(inputs.detach().cpu(), outputs.detach().cpu()):
                        if var_autoencoder:
                            out_img = out.view(nchans, width, height)
                        else:
                            out_img = out
                        avg_RMSE += (torch.sum((out_img-inp)**2))
                    autoencoder_str = '\n...average RMSE: %g' % (math.sqrt(avg_RMSE/ARGS.batch_size), )
                print '\nEpoch %.1f\n...loss: %g (%+.1f %% of loss at epoch %.1f)\n...time elapsed = %.0f s\n...lr = %g%s%s' % (
                    epoch + float(i)/len(dataloader), 
                    loss_avg, 
                    ((loss_avg-loss_min)/loss_avg)*100, 
                    epoch_loss_min, 
                    time.time()-t0_epoch, 
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    data_stats_str, autoencoder_str
                )
                
                # save statistics
                with open(path + '/.stats.txt', 'a') as stats_f: 
                    stats_f.write('%.1f %g %g\n' % (epoch + float(i)/len(dataloader), loss_avg, optimizer.state_dict()['param_groups'][0]['lr']))
                
                # update loss history
                losses.append(loss_avg)
                losses_ticks.append(epoch + float(i)/len(dataloader))
                if min(losses) < loss_min:
                    loss_min = min(losses)
                    epoch_loss_min = epoch + float(i)/len(dataloader)

                # save loss history plot
                plt.figure(figsize=(10, 5))
                plt.plot(losses_ticks, losses, 'b-', linewidth=1.0, aa=True, marker='.', ms=4.0)
                plt.hlines(loss_min, -1, ARGS.max_epochs+1, colors='b', linestyles='dashed')
                plt.xlim(0, ARGS.max_epochs+1)
                plt.xticks(np.arange(0, ARGS.max_epochs+1, step=5))
                plt.xlabel('Epochs')
                plt.ylabel('BCE + KL Loss' if var_autoencoder else type(criterion).__name__)
                plt.yscale('log')
                plt.savefig(path + '/loss.png', dpi=300)
                plt.close()

                # reset counters
                running_loss = 0.0
                t0_epoch = time.time()

            t0_data_load = time.time()
        
        total_time_epoch = time.time()-t0_epoch_loop
        print '\nEpoch %d completed in %.1f seconds' % (epoch+1, total_time_epoch)
        print '...of which forward required     %.1f %% of time' % ((total_time_forward/total_time_epoch)*100,)
        print '...of which backward required    %.1f %% of time' % ((total_time_backward/total_time_epoch)*100,)
        print '...of which optim equired        %.1f %% of time' % ((total_time_opt/total_time_epoch)*100,)
        print '...of which get batches required %.1f %% of time' % ((total_time_data_load/total_time_epoch)*100,)
        print '...of which other stuff required %.1f %% of time' % ((total_time_other/total_time_epoch)*100,)

        # save reconstructed images if autoencoder is used
        if conv_autoencoder or var_autoencoder:
            inputs_img = inputs.detach().cpu()
            outputs_img = outputs.detach().cpu()

            if var_autoencoder:
                outputs_img = outputs_img.view(outputs_img.shape[0], nchans, width, height)

            for inp, out in zip(inputs_img, outputs_img):
                inp = prepro.revert(inp)
                out = prepro.revert(out)

            plt.imsave(autoencoder_folder + '/original_' + str(epoch+1) + '.png', np.transpose(torchvision.utils.make_grid(inputs_img).numpy(), (1, 2, 0)))
            plt.imsave(autoencoder_folder + '/reconstructed_' + str(epoch+1) + '.png', np.transpose(torchvision.utils.make_grid(outputs_img).numpy(), (1, 2, 0)))
            plt.imsave(autoencoder_folder + '/error_' + str(epoch+1) + '.png', np.transpose(torchvision.utils.make_grid((outputs_img-inputs_img).abs_()).numpy(), (1, 2, 0)))

        # save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'net': net,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path + '/_epoch%04d.model.tar' % (epoch + 1))

    print '\nTraining COMPLETED in %.1f hours, experiment = %s' % ((time.time()-t0_train)/3600.0, os.path.basename(os.path.normpath(ARGS.workspace)))


# test function
def test(path, cross_validation_iter=None):

    print '\nTesting STARTED, experiment = %s' % (os.path.basename(os.path.normpath(ARGS.workspace)),)

    t0_test = time.time()
    test_batch_size = 512
    print '\nTesting parameters:'
    print '...device              = ' + str(DEVICE)
    print '...num workers         = ' + str(ARGS.num_workers)
    print '...batch size          = ' + str(test_batch_size)

    # load pretrained model
    models = sorted(glob.glob(path + '/_epoch*.tar'))
    if not models:
        raise Exception('no files matching _epoch*.tar pattern were found')
    print '...found %d checkpoints, load last epoch model %s' %(len(models), os.path.basename(models[-1]))
    checkpoint = torch.load(models[-1], map_location=lambda storage, loc: storage)
    net = checkpoint['net']
    print '...net                 = ' + type(net).__name__
    conv_autoencoder = False
    if "CAE" in ARGS.model:
        conv_autoencoder = True
    elif "VAE" in ARGS.model:
        #var_autoencoder = True
        raise NotImplementedError

    # instantiate preprocessing
    prepro = getattr(preprocessing, ARGS.preprocessing)()

    # load dataset
    dataset = load_dataset(False, (cross_validation_iter, ARGS.cross_valid_N), prepro)

    # instantiate data loader
    dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True)

    # load preprocessing parameters
    metadata_folder = path + "/.metadata"
    prepro.load(metadata_folder)

    # move net to cuda
    net.to(DEVICE)

    # preallocate scores arrays
    scores = []
    score_counter = np.zeros(dataset.n_classes, np.int32)
    for class_count in dataset.class_counts:
        scores.append(np.zeros(class_count, np.float64))

    # switch to test mode
    net.eval()

    # test samples
    print '\nTest samples'
    iters = len(dataloader)
    t0_test_loop = time.time()
    total_time_net = 0
    total_time_scores = 0
    total_time_data_load = 0
    t0_data_load = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 1):

            total_time_data_load += time.time() - t0_data_load

            if i%(iters/20) == 0:
                print '...iteration %d of %d, ETA = %.0f seconds' % (i, iters, (time.time()-t0_test_loop)*(iters-i)/i)
            
            # get the inputs
            inputs, labels = data
            
            # send to GPU
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # forward
            t0_net = time.time()
            outputs = net(inputs)
            total_time_net += time.time() - t0_net

            # get class scores from outputs
            if conv_autoencoder:
                t0_scores = time.time()
                for inp, out, label in zip(inputs.detach().cpu().numpy(), outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()):
                    scores[int(label)][score_counter[int(label)]] = math.sqrt(np.sum((out-inp)**2))
                    score_counter[int(label)] += 1
                total_time_scores += time.time() - t0_scores
            else:
                t0_scores = time.time()
                for output, label in zip(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()):
                    scores[int(label)][score_counter[int(label)]] = output[1]
                    score_counter[int(label)] += 1
                total_time_scores += time.time() - t0_scores

            t0_data_load = time.time()
        print '...total time elapsed      = %.0f seconds' % (time.time()-t0_test_loop,)
        print '...of which test net is    = %.0f seconds' % total_time_net
        print '...of which get scores is  = %.0f seconds' % total_time_scores
        print '...of which get batches is = %.0f seconds' % total_time_data_load

    # save outputs
    for class_i in range(dataset.n_classes):
        with open(ARGS.workspace + "/%d-fold-image-based-class%d.sco" % (ARGS.cross_valid_N, class_i), "w" if cross_validation_iter == 1 else "a") as scores_f:
            if cross_validation_iter == 1:
                scores_f.write("#SAMPLE 	#SCORE\n")
            for idx, score in enumerate(scores[class_i]):
                scores_f.write('%d\t%f\n' % (idx, score)) 

    print '\nTesting COMPLETED in %.1f hours, experiment = %s' % ((time.time()-t0_test)/3600.0, os.path.basename(os.path.normpath(ARGS.workspace)))


# crossvalid function
def crossvalid():

    t0_cv = time.time()
    print '\n%d-fold image-based cross-validation STARTED' % ARGS.cross_valid_N
    print '...fold start = %d' % ARGS.fold_start
    print '...fold end   = %d' % (ARGS.cross_valid_N if ARGS.fold_end < 0 else ARGS.fold_end)
    for i in range(ARGS.fold_start, ARGS.cross_valid_N+1 if ARGS.fold_end < 0 else ARGS.fold_end + 1):
        
        t0_cv_i = time.time()
        print '\nCross-validation iteration %d in [%d,%d] STARTED' % (i, 1, ARGS.cross_valid_N)
        
        # create output folder if it does not exist
        cross_valid_folder = ARGS.workspace + "/%d-fold%02d-image-based" % (ARGS.cross_valid_N, i)
        if not os.path.exists(cross_valid_folder):
            os.makedirs(cross_valid_folder)
        
        # train and test
        if not ARGS.skip_training:
            train(cross_valid_folder, i)
        if not ARGS.skip_testing:
            test(cross_valid_folder, i)

        print '\nCross-validation iteration %d in [%d,%d] COMPLETED in %.0f seconds' % (i, 1, ARGS.cross_valid_N, time.time()-t0_cv_i)
    print '\n%d-fold image-based cross-validation COMPLETED in %.0f seconds\n\n\n' % (ARGS.cross_valid_N, time.time()-t0_cv)

    # call rocalc to normalize scores and get ROC curve
    command = "rocalc --sco2fpr --pos_file=\"%s/%d-fold-image-based-class1.sco\" --neg_file=\"%s/%d-fold-image-based-class0.sco\" --auto" % (ARGS.workspace, ARGS.cross_valid_N, ARGS.workspace, ARGS.cross_valid_N)
    os.system(command)
    command = "rocalc --roc --pos_file=\"%s/%d-fold-image-based-class1.fpr\" --neg_file=\"%s/%d-fold-image-based-class0.fpr\" --output=\"%s/%d-fold-image-based-ROC.small.txt\" --small --averaged -j=%d --iters=1000" % (ARGS.workspace, ARGS.cross_valid_N, ARGS.workspace, ARGS.cross_valid_N, ARGS.workspace, ARGS.cross_valid_N, multiprocessing.cpu_count())
    os.system(command)

def main():

    # parse command-line arguments
    global ARGS
    ARGS = PARSER_GLOBAL.parse_args()

    # select CUDA device
    # @FIXED by Alessandro on 2018-09-06: hide all cuda devices except the selected one
    # so that PyTorch will not allocate 160 MB of RAM on cuda device 0 (bug #7392 in v0.4)
    os.environ["CUDA_VISIBLE_DEVICES"] = repr(ARGS.gpu)
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create workspace folder if it does not exist
    if not os.path.exists(ARGS.workspace):
        os.makedirs(ARGS.workspace)

    # save command file
    command_file_path = '%s/commands.%s.txt' % (ARGS.workspace, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S'))
    with open(command_file_path, 'w') as command_file:
        command_file.write('%s' % ' '.join(sys.argv[1:]))

    # launch the selected modality
    if ARGS.command == 'train':
        train(ARGS.workspace)
    elif ARGS.command == 'test':
        test(ARGS.workspace)
    else:
        crossvalid()

if __name__ == '__main__':
    main()
