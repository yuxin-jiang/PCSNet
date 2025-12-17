<div align="left">
<br>
<br>
</div>
<div align="center">
<h1>PCSNet: Prototypical Learning Guided Context-Aware Segmentation Network for Few-Shot Anomaly Detection</h1>
<div>
&nbsp;&nbsp;&nbsp;&nbsp;Yuxin Jiang<sup>1</sup>&emsp;
&nbsp;&nbsp;&nbsp;&nbsp;Yunkang Cao<sup>1 </sup>&emsp;
&nbsp;&nbsp;&nbsp;&nbsp;Weiming Shen<sup>1, *</sup>&emsp;
</div>
<div>
&nbsp;&nbsp;&nbsp;&nbsp;<sup>1</sup>Huazhong University of Science and Technology
</div>
  
[[Paper]](https://ieeexplore.ieee.org/document/10702559/)
  
![Framework](Figure/figure1.png)

---

</div>

>**Abstract:** Few-shot anomaly detection (FSAD) denotes the identification of anomalies within a target category with a limited number of normal samples. While pre-trained feature representations play an important role in existing FSAD methods, there exists a domain gap between pre-trained representations and target FSAD scenarios. This study proposes a Prototypical Learning Guided Context-Aware Segmentation Network (PCSNet) to address the domain gap and improve feature descriptiveness in target scenarios. In particular, PCSNet comprises a prototypical feature adaption (PFA) sub-network and a context-aware segmentation (CAS) sub-network. PFA extracts prototypical features as accurate guidance to ensure better feature compactness for normal data while distinct separation from anomalies. A pixel-level disparity classification loss is also designed to make subtle anomalies more distinguishable. Then a CAS sub-network is introduced for pixel-level anomaly localization, where pseudo anomalies are exploited to facilitate the training process. Experimental results on MVTec and MPDD demonstrate the superior FSAD performance of PCSNet, with 94.9% and 80.2% image-level AUROC in an 8-shot scenario, respectively. Real-world applications on automotive plastic part inspection further demonstrate that PCSNet can achieve promising results with limited training samples.

**Index Terms:** Anomaly detection; Pretrained feature representations; Few-shot learning; Prototypical learning;

## 💻 Requirements
- pytorch == 1.12.0
- torchvision == 0.13.0
- numpy == 1.21.6
- scipy == 1.7.3
- matplotlib == 3.5.2
- tqdm
  
## 🛠️ Framework
![Framework](Figure/figure2.png)

## 📥 Dataset
Download the MVTec dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).

## 🚀 Usage
Execute the following command for training and evaluation:
```bash
python main.py
```

## 📊 Quantitative Comparisons

PCSNet demonstrates competitive performance against other few-shot anomaly detection (FSAD) methods (such as RD4AD, CFA, PatchCore, RegAD, RFR, and PACKD).

- On the MVTec AD dataset for image-level anomaly detection (AUROC):
  - Outperforms RegAD by 4.7%, 3.9%, and 3.7% in the 2-shot, 4-shot, and 8-shot settings, respectively.
  - Outperforms CFA by 9.3%, 7.1%, and 4.0% in the same settings.

- On the MPDD dataset, achieves 80.2% image-level AUROC in the 8-shot setting, surpassing PACKD by 9.7% and RegAD by 8.3%.

- For anomaly localization (pixel-level AUROC), PCSNet ranks first or second on both datasets.

![Results](https://github.com/yuxin-jiang/PCSNet/blob/main/Figure/AD_result.png)

## 🖼️ Qualitative Visualization (Anomaly Localization)

Visualization results show that PCSNet excels in anomaly localization. It accurately captures large anomalous regions (e.g., hazelnut, cable), precisely localizes small anomalies (e.g., capsule, pill, screw), and detects all multiple anomalous regions without omission (e.g., grid, toothbrush).

In contrast, other methods often produce coarse localization or misclassify normal regions, while PCSNet provides significantly more precise anomaly maps.

![Anomaly Detection and Localization Results](https://github.com/yuxin-jiang/PCSNet/blob/main/Figure/AD_result.png)


## 📝 Citation
If you find this work useful, please consider citing:
```
@article{jiang2024prototypical,
  title={Prototypical learning guided context-aware segmentation network for few-shot anomaly detection},
  author={Jiang, Yuxin and Cao, Yunkang and Shen, Weiming},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```
