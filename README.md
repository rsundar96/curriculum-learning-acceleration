# Curriculum Learning Acceleration

The aim of this project is to investigate whether using a Curriculum Learning training strategy helps speed up the training convergence and model generalisation of various CNNs (VGGNet and ResNet) on Image Recognition tasks. The code used to obtain the results as given in my Master's Thesis is present here, along with instructions on how to execute the code. Structure of the repository is as follows:

**Keras Curriculum Learning**: Keras code used to replicate the results as detailed in [On The Power of Curriculum Learning in Training Deep Networks](https://arxiv.org/pdf/1904.03626.pdf).

**PyTorch-CIFAR**: Repositories containing the PyTorch implementations of the various experiments conducted on the CIFAR-10 and CIFAR-100 datasets.

## Code Execution

### Keras Curriculum Learning

Refer to the instructions given in [Guy Hacohen's repository](https://github.com/GuyHacohen/curriculum_learning). This should enable you to replicate the baseline results.

### PyTorch CIFAR

#### I. CIFAR-10

Run `python main.py` in order to start the training process. Supports VGG19 and ResNet50. Training may also be resumed by running `python main.py --resume --lr=0.01`.

#### II. CIFAR-100

Run `python train.py -net [MODEL] -gpu`, where `[MODEL]` can be any of the vgg (vgg11/vgg13/vgg19) or resnet (resnet18/resnet50) based variants. The weights file with thee best accuracy will be written to disk with the suffix "best" (located in the checkpoint folder).

To test the model, run `python test.py -net [MODEL] path_to_[MODEL]_weights_file`.

### Code References

- Guy Hacohen Curriculum Learning [repository](https://github.com/GuyHacohen/curriculum_learning)
- kuangliu PyTorch CIFAR-10 [repository](https://github.com/kuangliu/pytorch-cifar)
- weiaicunzai PyTorch CIFAR-100 [repository](https://github.com/weiaicunzai/pytorch-cifar100)
