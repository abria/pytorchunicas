# pytorchunicas
A PyTorch wrapper for unbalanced training/classification on image rois

```
usage: pytorchunicas.py [-h] <command> ...

optional arguments:
  -h, --help  show this help message and exit

subcommands:

  <command>    description

  train        Train or finetune a model
  test         Score a model
  crossvalid   Cross-validation (includes multiple iterations of training and
               testing)
```

Example of usage for the `crossvalid` modality (includes both training and testing phases using k-fold cross-validation):

```
usage: pytorchunicas.py crossvalid [-h] --model
                                   {CAEP16,CAEP23L2,CAEP23L2F2,CAEP23L3,CAEP48,CAE_MNIST,VAE_MNIST,VGGnetP12L2,VGGnetP16L2,VGGnetP16L2BN,VGGnetP23L2,VGGnetP23L3}
                                   --roi_files ROI_FILES --img_folder
                                   IMG_FOLDER --workspace WORKSPACE
                                   [-b BATCH_SIZE] [--gpu GPU]
                                   [--img_prefix IMG_PREFIX]
                                   [--img_suffix IMG_SUFFIX]
                                   [--img_list IMG_LIST]
                                   [--img_channel IMG_CHANNEL]
                                   [--max_epochs MAX_EPOCHS]
                                   [--class_weights CLASS_WEIGHTS]
                                   [--class_max_counts CLASS_MAX_COUNTS]
                                   [--class_min_counts CLASS_MIN_COUNTS]
                                   [--preprocessing {ImageStandardization,MinMaxNormalization,NoPreprocessing,PixelStandardization}]
                                   [--augmentation {FlipRotate,Replicate}]
                                   [--optimizer {ASGD,Adadelta,Adagrad,Adam,Adamax,LBFGS,RMSprop,Rprop,SGD,SparseAdam}]
                                   [--lr_base LR_BASE]
                                   [--lr_stepsize LR_STEPSIZE]
                                   [--lr_gamma LR_GAMMA] [--momentum MOMENTUM]
                                   [--weight_decay WEIGHT_DECAY]
                                   [--display DISPLAY]
                                   [--loss_weights LOSS_WEIGHTS] [--resume]
                                   [--start_epoch N] [-j NUM_WORKERS]
                                   [--no_reshuffle] [--data_stats]
                                   [--cross_valid_N CROSS_VALID_N]
                                   [--fold_start FOLD_START]
                                   [--fold_end FOLD_END] [--skip_training]
                                   [--skip_testing]

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        mini-batch size (default: 32)
  --gpu GPU             GPU id to use.
  --img_prefix IMG_PREFIX
                        Prefix to be added to all the images. (default: )
  --img_suffix IMG_SUFFIX
                        Suffix to be added to all the images. (default: )
  --img_list IMG_LIST   Image inclusion list (only load samples if they belong
                        to the images contained in this list).
  --img_channel IMG_CHANNEL
                        Image channel selection according to BGR color space
                        indexing (B = 0, G = 1, R = 2).
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train or test. In the
                        latter case, this determines the model to load.
                        (default: 30)
  --class_weights CLASS_WEIGHTS
                        Class weights determining the class sampling
                        probability at training time.
  --class_max_counts CLASS_MAX_COUNTS
                        Max number of samples that can be loaded, for each
                        class.
  --class_min_counts CLASS_MIN_COUNTS
                        Min number of samples that can be loaded, for each
                        class.
  --preprocessing {ImageStandardization,MinMaxNormalization,NoPreprocessing,PixelStandardization}
                        Data preprocessing. (default: PixelStandardization)
  --augmentation {FlipRotate,Replicate}
                        Data augmentation. (default: FlipRotate)
  --optimizer {ASGD,Adadelta,Adagrad,Adam,Adamax,LBFGS,RMSprop,Rprop,SGD,SparseAdam}
                        Optimizer. (default: SGD)
  --lr_base LR_BASE     Base learning rate. (default: 0.001)
  --lr_stepsize LR_STEPSIZE
                        Period of learning rate decay (in epochs). (default:
                        6)
  --lr_gamma LR_GAMMA   Multiplicative factor of learning rate decay.
                        (default: 0.1)
  --momentum MOMENTUM   Momentum factor. (default: 0.9)
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 penalty). (default: 0)
  --display DISPLAY     The number of epochs between displaying info.
                        (default: 0.1)
  --loss_weights LOSS_WEIGHTS
                        Class weights assigned when computing the loss
                        function (weights < 0 will be assigned automatically)
  --resume              Resume training from last epoch model. (default:
                        False)
  --start_epoch N       Manual epoch number (useful on restarts) (default: 0)
  -j NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of workers for mini batch loading (default: 2)
  --no_reshuffle        Disables rehuffling of training samples at the
                        beginning of each epoch. (default: False)
  --data_stats          Print data statistics (min, max, mean, std) during
                        training (default: False)
  --cross_valid_N CROSS_VALID_N
                        Number of cross-validation folds. (default: 2)
  --fold_start FOLD_START
                        Fold start (default: 1)
  --fold_end FOLD_END   Fold end (default: -1)
  --skip_training       Skip training phase. (default: False)
  --skip_testing        Skip testing phase. (default: False)

required named arguments:
  --model {CAEP16,CAEP23L2,CAEP23L2F2,CAEP23L3,CAEP48,CAE_MNIST,VAE_MNIST,VGGnetP12L2,VGGnetP16L2,VGGnetP16L2BN,VGGnetP23L2,VGGnetP23L3}
                        Model architecture. (default: None)
  --roi_files ROI_FILES
                        Roi files separated by ',' one for each class.
  --img_folder IMG_FOLDER
                        Directory with all the images.
  --workspace WORKSPACE
                        Folder containing input and output files.
```
