# Cross-view Consistent Attention: Transformer-based Geo-localization for UAV and Satellite Images
## dataset

download the dataset see : https://github.com/layumi/University1652-Baseline

## Training

```
python -u mytrain.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 --epochs 50 --save_path ./university_exp66 --op adamw --wd 5e-4 --mining --dataset university1652 --cos --asam --rho 2.5 --asam --gpu 0  --dim 1000 --dim_out 1000 --loss_func euclid
```

## Evaluation

```
python -u validation.py --lr 0.00001 --batch-size 32 --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 --epochs 50 --save_path ./university_exp66 --op adamw --wd 5e-4 --mining --dataset university1652 --cos --asam --rho 2.5 --asam --gpu 0  --dim 1000 --dim_out 1000 --loss_func euclid --resume '/root/transgeo/model_best.pth.tar'  -e
```
