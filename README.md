# PCSNet - Prototypical Learning Guided Context-Aware Segmentation Network for Few-Shot Anomaly Detection

This is an official implementation of “ Prototypical Learning Guided Context-Aware Segmentation Network for Few-Shot Anomaly Detection” (PCSNet) with PyTorch.

![](https://github.com/yuxin-jiang/PCSNet/blob/main/Figure/figure1.png)
**Abstract:** Few-shot anomaly detection (FSAD) denotes the identification of anomalies within a target category with a limited number of normal samples. While pre-trained feature representations play an important role in existing FSAD methods, there exists a domain gap between pre-trained representations and target FSAD scenarios. This study proposes a Prototypical Learning Guided Context-Aware Segmentation Network (PCSNet) to address the domain gap and improve feature descriptiveness in target scenarios. In particular, PCSNet comprises a prototypical feature adaption (PFA) sub-network and a context-aware segmentation (CAS) sub-network. PFA extracts prototypical features as accurate guidance to ensure better feature compactness for normal data while distinct separation from anomalies. A pixel-level disparity classification loss is also designed to make subtle anomalies more distinguishable. Then a CAS sub-network is introduced for pixel-level anomaly localization, where pseudo anomalies are exploited to facilitate the training process. Experimental results on MVTec and MPDD demonstrate the superior FSAD performance of PCSNet, with 94.9% and 80.2% image-level AUROC in an 8-shot scenario, respectively. Real-world applications on automotive plastic part inspection further demonstrate that PCSNet can achieve promising results with limited training samples. 

**Index Terms:** Anomaly detection; Pretrained feature representations; Few-shot learning; Prototypical learning; 

# Implementation
1. Environment.<br />
>pytorch == 1.12.0

>torchvision == 0.13.0

>numpy == 1.21.6

>scipy == 1.7.3

>matplotlib == 3.5.2

>tqdm

2. Dataset.<br />
>Download the MVTec dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).<br />

3. Execute the following command to see the training and evaluation results.<br />
```
python main.py
```
# Results

| AUC(%) | 2 | 4 | 8 |
| :---: | :---:  | :---:  | :---:  |
| bottle | 	 99.8±0.1  | 99.9±0.2| 100.0±0.0	|
|cable	|	89.7±1.9	| 91.0±3.3 	| 93.9±0.2	|
|capsule	| 72.3±8.9 |	76.5±10.5| 88.6±1.5	|
|carpet	| 99.3±0.3		| 98.9±0.5		| 98.5±0.5	| 
|grid		| 93.7±2.2 | 92.8±2.2	| 97.9±0.4	|
|hazelnut	|	95.1±2.0	| 98.6±2.0| 99.9±0.1	| 
|leather |100.0±0.0 |	100.0±0.0 	| 100.0±0.0	| 
|metal_nut	| 88.7±3.0 |	93.7±5.8| 96.5±1.3	| 
|pill	| 87.3±1.6	| 90.6±1.8| 87.9±0.8 |	
|screw	 |	48.2±2.5	| 56.2±7.5 |	65.3±2.9	|
|tile |	98.5±0.2	| 99.0±0.3	| 99.3±0.1 |	
|toothbrush	| 87.8±0.7 	| 90.6±5.4	 |	98.3±0.2	| 
|transistor |	99.3±0.2 | 95.9±6.2	| 99.7±0.1	| 
|wood |	98.7±0.6 | 99.1±0.3		| 99.0±0.1	| 
|zipper	 |	98.3±0.7	| 99.0±0.4	| 98.8±0.2	| 
|average		| 90.4±0.6	| 92.1±0.9 |94.9±0.2 |

# Visualization

![](https://github.com/yuxin-jiang/PCSNet/blob/main/Figure/Result.png)

# Reference
```
