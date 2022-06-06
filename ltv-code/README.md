# LTV-prediction

This repo is the python implementation of paper "Learning Reliable User Representations from Volatile and Sparse Data to Accurately Predict Customer Lifetime Value" (KDD2021).

### Preliminary
Please first install the following python packages:

pytorch  
DGL  
pandas  
numpy  
scikit-learn  
matplotlib  
tqdm  
statsmodels  



### Run
`cd ltv-code/src/`

`python trainer.py -c <config file path> -d <indice of GPU, e.g., 0>`



### Data
Due to the data privacy restrictions, we have deleted some data preprocessing details with sensitive information in data_prepare.py.
Please use your own dataset and customized dataset reader.


### Citation
If you find our paper or code is useful for you, please cite the following BibTex:

> @article{Xing2021LearningRU,  
  title={Learning Reliable User Representations from Volatile and Sparse Data to Accurately Predict Customer Lifetime Value},  
  author={Mingzhe Xing and Shuqing Bian and Wayne Xin Zhao and Zhen Xiao and Xinji Luo and Cunxiang Yin and Jing Cai and Yancheng He},  
  journal={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},  
  year={2021}  
}  
