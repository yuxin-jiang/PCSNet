# PCSNet - Prototypical Learning Guided Context-Aware Segmentation Network for Few-Shot Anomaly Detection

This is an official implementation of “A Masked Reverse Knowledge Distillation Method Incorporating Global and Local Information for Image Anomaly Detection” (MRKD) with PyTorch, accepted by knowledge-based systems.

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
|Data	|Bottle |	Cable	| Capsule	| Carpet	| Grid	| Hazelnut	| Leather |	Metalnut	| Pill	| Screw	| Tile |	| Tooth. |	| Transistor |	| Wood |	| Zipper |	Ave.|
| ------------- | ------------- ||
|2|99.8±0.1|	89.7±1.9	|72.3±8.9	|99.3±0.3	|93.7±2.2|	95.1±2.0	|100.0±0.0	|88.7±3.0	|87.3±1.6|	48.2±2.5|	98.5±0.2	|87.8±0.7|	99.3±0.2|	98.7±0.6|	98.3±0.7	|90.4±0.6
|4|99.9±0.2	|91.0±3.3|	76.5±10.5	|98.9±0.5	|92.8±2.2	|98.6±2.0|	100.0±0.0|	93.7±5.8	|90.6±1.8	|56.2±7.5	|99.0±0.3	|90.6±5.4	|95.9±6.2	|99.1±0.3	|99.0±0.4	|92.1±0.9|
|8|100.0±0.0	|93.9±0.2	|88.6±1.5	|98.5±0.5	|97.9±0.4	|99.9±0.1	|100.0±0.0	|96.5±1.3	|87.9±0.8|	65.3±2.9	|99.3±0.1|	98.3±0.2	|99.7±0.1	|99.0±0.1	|98.8±0.2	|94.9±0.2|

# Reference
```
