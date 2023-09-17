# Breaking Time Invariance: Assorted-Time Normalization for RNNs paper Supplementary Material

This repository contains the code used in Breaking Time Invariance: Assorted-Time Normalization for RNNs paper.

Required packages:
+ python==3.7 and CUDA 9.0
+ torch==1.1.0
+ numpy==1.14.5
+ tqdm==4.60.0

To run adding problem with T=100 for LSTM/LN/ATN model:
+ `python adding_problem.py --mode lstm`
+ `python adding_problem.py --mode ln`
+ `python adding_problem.py --mode atn`

To run copying problem with T=100 for:
+ LSTM
  ```sh
  python copying_problem.py --mode lstm
  ```
+ LN
  ```sh
  python copying_problem.py --mode ln
  ```
+ ATN (ours)
  ```sh
  python copying_problem.py --mode atn
  ```

To run character language modeling with Penn Treebank (PTB) dataset for LN/ATN model:
`cd ptb` and run `getdata.sh` to acquire the Penn Treebank dataset
+ `python main.py --max_length 1` - for LN
+ `python main.py` - for ATN

In addition, we provide the ATN-LSTM file `atn_lstm.py` that has implementation of our method.

If you use this code or our results in your research, please cite as appropriate:

```
@misc{https://doi.org/10.48550/arxiv.2209.14439,
  doi = {10.48550/ARXIV.2209.14439},
  url = {https://arxiv.org/abs/2209.14439},
  author = {Pospisil, Cole and Zadorozhnyy, Vasily and Ye, Qiang},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Breaking Time Invariance: Assorted-Time Normalization for RNNs},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## Acknowledgments
This project is partially built on top of 
+ [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm) for PTB problem
+ [expRNN](https://github.com/Lezcano/expRNN) for copying problem
