# Improving Robustness of Computer Vision Models via Different Adversarial Training

**Improving Robustness of Computer Vision Models via Different Adversarial Training**\
Yuchen Xu, Junjie Wang, Tianzhen Yang, Wenxuan Zhang\
_Computer Vision PBL Research_

## Overview and Key Ideas
Our method is based on the FreeAT algorithm for improvement. By evaluating and adjusting the value of parameter m in real-time during the training process, we achieve better training results, resulting in higher adversarial accuracy and cleaning accuracy while maintaining training time that is basically the same as the original method.

## Free-Adapt Algorithm
Under working

## Performance Comparisons
Under working

## Running Experiments

### Requirements

```sh
pip install -r requirements.txt
```

### Training Standard Models
Any experiments on cifar10 can be run with the `--pretrained` flag in order to intialize with these weights as was done in the paper. In order to train a standard model on cifar10 run:

```
python train.py --mode=standard --dataset=cifar10 --data_path=./data --num_epochs=160 --ep_decay=60
```

### Training Free-Adapt

We provide ready-to-run code for our Free-Adapt model and the following fast AT baselines: PGD, FGSM-Adapt, FGSM-GA, Free. See details in our paper. Use the following:

```sh
python train.py --mode=free_adapt --min_dr=r \
    --data_path=./data \
    --lr=0.1 \
    --num_epochs=30 \
    --ep_decay=15
    --pretrained
```

To run Free-Adapt-AT with L-infinity norm `8/255` at distortion ratio `r` on CIFAR-10 with data located in `./data`, for `30` epochs with an initial SGD learning rate of `0.1` that decays by 0.1 every `15` epochs. The `--pretrained` flag assumes you have a pretrained model named:

```yaml
cifar10:  {YOUR_PATH}\\pretrained_model\\standard\\checkpoints\\checkpoint__best.pt
cifar100: <see `Training Standard Models`>
```

this flag may be removed and the model will train from scratch.

Details about the various modes and flags and AT modes can be found in `AdversarialTrainer.AdversarialTrainer.__init__`. 

Results are saved in:
```
|-topdir/
    |-exp_tag/
        |-checkpoints/
            |-<model checkpoints>.pt
        |-hparams.yaml (params used in trainer)
        |-train_results.csv
        |-eval_results.csv 
```

### Training Other Baselines
We provide some basic examples of training other AT methods at epsilon=8/255. For more details please see `AdversarialTrainer.AdversarialTrainer.__init__`. All methods have the flags

```
    --data_path=./data \
    --lr=0.1 \
    --num_epochs=30 \
    --ep_decay=15 \
    --pretrained
```

passed after them as was done in Free-Adapt above.

**PGD** with 7 steps
```
python train.py --mode=pgd --K=7 \
```

**FGSM-Adapt** with 4 checkpoints
```
python train.py --mode=fgsm_adapt --K=4 \
```

**FGSM-GA** with regularizer 0.5
```
python train.py --mode=grad_align --grad_align_lambda=0.5 \
```

**Free** with 8 minibatch replays
```
python train.py --mode=free --K=8
```


### AutoAttack and PGD-50 Evaluation

Install autoattack into your visual environment.
```
pip install git+https://github.com/fra31/auto-attack
```

After training the model you can evaluate with AutoAttack and PGD-50 via

```
python autoattack_eval.py --exp_dir=topdir/exp_tag/ --data_path=./data
```

Note these attacks are performed using the epsilon value chosen in `hparams.yaml`. The results will be in
```
|-topdir/
    |-exp_tag/
        |-autoattack_and_pgd50_value.yaml
```

## License

The code for Free-Adapt is licensed under the MIT License.

## References

> Tsiligkaridis, T., & Roberts, J. (2022). Understanding and increasing efficiency of Frank-Wolfe adversarial training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 50-59).

> Shafahi, A., Najibi, M., Ghiasi, M. A., Xu, Z., Dickerson, J., Studer, C., ... & Goldstein, T. (2019). Adversarial training for free!. Advances in Neural Information Processing Systems, 32.


## Acknowledgement

Under working
