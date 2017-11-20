# Enc-Dec-AD
Anormaly Detection With Deep Learning Encoder Decoder(https://arxiv.org/abs/1607.00148?context=cs)
![EncoderDecoder](https://github.com/misogil0116/Enc-Dec-AD/blob/cf0ff4cbb45ba4c0e426906901694c7d2eb889e4/image/encdec.png "image")
# Usage
you make directory like this
```
Enc-Dec-AD
    - .gitignore
    - EncDecAD.py
    - README.md
    - data
        - anormaly_data
            - train_and_test
                - train.npy
                - test.npy
    - model
```

if you want to change the train source path and test source path,

you can change it when you make instance of EncDecAD class.

```python
model = EncDecAD("train_source", "test_source")
```

# [WIP]

- with Attention
- add batch process mode for GPU(branch batch_calculation, but something wrong..)
- calculation of Gaussian params μ and Σ
