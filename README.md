# ASFormer: Transformer for Action Segmentation
This repo provides training &amp; inference code for BMVC 2021 paper: [ASFormer: Transformer for Action Segmentation](https://arxiv.org/pdf/2110.08568.pdf) 

# Update 
Thanks to @ddz16 . There is a small error in the code (L72, model.py), but it does not affect the main conclusions of this paper or the performance, see [#2](https://github.com/ChinaYi/ASFormer/issues/2). In order to load the pre training model of the paper, we do not update the repo to fix the bug. However, we still suggest to make a manual change after downloading the code.

```
(L72, model.py) window_mask[:, :, i:i+self.bl] = 1  change to window_mask[:, i, i:i+self.bl] = 1
```

## Enviroment
Pytorch == 1.1.0, torchvision == 0.3.0, python == 3.6, CUDA=10.1

## Reproduce our results
```
1. Download the dataset data.zip at (https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) or (https://zenodo.org/record/3625992#.Xiv9jGhKhPY). 
2. Unzip the data.zip file to the current folder. There are three datasets in the ./data folder, i.e. ./data/breakfast, ./data/50salads, ./data/gtea
3. Download the pre-trained models at (https://pan.baidu.com/s/1zf-d-7eYqK-IxroBKTxDfg) or (https://drive.google.com/file/d/1xNykN3vXMHCpHIYT0eb5ZHnKSu3Y2K8r/view?usp=sharing). There are pretrained models for three datasets, i.e. ./models/50salads, ./models/breakfast, ./models/gtea
4. Run python main.py --action=predict --dataset=50salads/gtea/breakfast --split=1/2/3/4/5 to generate predicted results for each split.
5. Run python eval.py --dataset=50salads/gtea/breakfast --split=0/1/2/3/4/5 to evaluate the performance. **NOTE**: split=0 will evaulate the average results for all splits, It needs to be done after you complete all split predictions.
```

## Train your own model
Also, you can retrain the model by yourself with following command.
```
python main.py --action=train --dataset=50salads/gtea/breakfast --split=1/2/3/4/5
```
The training process is very stable in our experiments. It convergences very fast and is not sensitive to the number of training epochs.


## Demo for using ASFormer as your backbone
In our paper, we replace the original TCN-based backbone model [MS-TCN](https://github.com/yabufarha/ms-tcn) in [ASRF](https://github.com/yiskw713/asrf) with our ASFormer.  The new model achieves even higher results on the 50salads dataset than the original ASRF. [Code is Here](https://github.com/ChinaYi/asrf_with_asformer).


------
If you find our repo useful, please give us a star and cite
```
@inproceedings{chinayi_ASformer,  
	author={Fangqiu Yi and Hongyu Wen and Tingting Jiang}, 
	booktitle={The British Machine Vision Conference (BMVC)},   
	title={ASFormer: Transformer for Action Segmentation},
	year={2021},  
}
```
Feel free to raise a issue if you got trouble with our code.
