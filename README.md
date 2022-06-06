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
If you find our paper or code is useful for your research work, please cite the following BibTex:

> @inproceedings{xing2021learning,  
  title={Learning Reliable User Representations from Volatile and Sparse Data to Accurately Predict Customer Lifetime Value},  
  author={Xing, Mingzhe and Bian, Shuqing and Zhao, Wayne Xin and Xiao, Zhen and Luo, Xinji and Yin, Cunxiang and Cai, Jing and He, Yancheng},   
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},  
  pages={3806--3816},  
  year={2021}  
}
