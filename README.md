# Breaking Time Invariance: Assorted-Time Normalization for RNNs paper Supplementary Material

This repository contains the code used in the <a target="_blank" href="https://arxiv.org/abs/2209.14439">Breaking Time Invariance: Assorted-Time Normalization for RNNs paper</a>.

Required packages:
+ CUDA 9.0
+ python==3.7
+ torch==1.1.0
+ numpy==1.14.5
+ tqdm==4.60.0

To run adding problem with T=100:
+ LSTM
  ```sh
  python adding_problem.py --mode lstm
  ```
+ LN
  ```sh
  python adding_problem.py --mode ln
  ```
+ ATN (ours)
  ```sh
  python adding_problem.py --mode atn
  ```

To run copying problem with T=100:
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

To run character language modeling with Penn Treebank (PTB) dataset for LN/ATN model. First, go to `ptb` folder and acquire the Penn Treebank dataset:
```sh
cd ptb
bash getdata.sh
```
Then 
+ LN
```sh
  python main.py --max_length 1
```
+ LSTM
```sh
  python main.py
```

In addition, we provide the ATN-LSTM file `atn_lstm.py` that has the implementation of our method.

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
