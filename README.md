# ICML2019 Proportionally Fair Clustering



Authors: Xingyu Chen, Brandon Fain, Charles Lyu, Kamesh Munagala



This repository serves as the code base for ICML2019 Submission Paper: Proportionally Fair Clustering.



#### Requirements:



Python >= 3.7.0



scikit-learn >= 0.20.0



matplotlib >= 3.0.2



numpy >= 1.15.4



#### How to repeat our experiments

Go to root folder, run `python experiment.py`  to start experiment. In command line, you can specify the <img src="https://latex.codecogs.com/gif.latex?\rho " /> proportionality measure used for local capture algorithm. You can also specify which database you run your experiment with `--file_name` option. (available names `kdd` , `iris` , `diabetes` ). To repeat the three experiments we did, you can run the following three commands,



```

python experiment.py --file_name iris --rho 1.00001

python experiment.py --file_name diabetes --rho 1.00001

python experiment.py --file_name kdd --rho 2 --sample

```



To customize your own experiments, see More Help section for support.



#### More Help



usage: experiment.py [-h] [--sample] [--sample_clients SAMPLE_NUM]

​                     [--sample_centers CENTER_NUM] [--file_name FILE_NAME]

​                     [--rho RHO]



optional arguments:

  -h, --help            show this help message and exit

  --sample

  --sample_clients SAMPLE_NUM

  --sample_centers CENTER_NUM

  --file_name FILE_NAME

  --rho RHO