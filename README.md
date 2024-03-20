## Introduction
PyTorch implementation of Pre-LayerNorm Transformer architecture ([Xiong et al. 2020](https://arxiv.org/pdf/2002.04745.pdf)). Developed architecture was trained on [Multi30k dataset](https://github.com/multi30k/dataset) for English to German translation. This dataset is small - only ~30k sentence pairs in the training set, as this project served for educational purposes primarily. 

## Data preprocessing
First, basic preprocessing techniques are employed - removal of repeated whitespaces, replacing all parentheses with (,), replacing all quotes with double quotes, lowercasing. Afterwards, independent BPEs are trained on English and German sentences (credit to fast BPE library [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe)) - maximum vocabulary size was set to 30k, which due to shortage of data led to English BPE having a vocabulary size of 20657, in some sense it compressed much better than German. Both vocabularies include special tokens for start of sequence, end of sequence, padding tokens etc. For a fixed tokenizer, tokenization of text is a deterministic task, so after training of BPEs is done both English and German sentences are pre-tokenized and saved to a file to improve performance. For our custom datasets which extends from torch.utils.Dataset, we've used linecache library to avoid reading entire datasets into memory.

## Detailed model description
Trained model had the following hyperparameters:
| Parameter | Value |
| -------- | -------- |
| $d_{model}$ | 512 |
| $d_{ff}$ | $4 \times 512 = 2048$|
| Number of layers | 6 | 
| Attention heads | 8 | 
| $d_{k}$ | $\frac{512}{8} = 64$|
| $d_{v}$ | $d_{k}$ |
| $p_{dropout}$ | 0.1 |

Notation is the same as in [Vaswati et al. 2017](https://arxiv.org/pdf/1706.03762.pdf). In addition, dropout was also applied to attention weights. Positional embeddings are used instead of fixed wave embeddings as in the previous paper. Since tokenizers differ for source and target language, different token embedding tables in encoder and decoder are used, along with different positional embedding tables. This resulted in a model of $\approx85M$ parameters.

## Training
Loss function of choice was label-smoothed cross-entropy with $\epsilon_l = 0.1$. Optimizer of choice was Adam, with an initial learning rate of $5e-4$ which was scaled by $0.95$ every 5-th epoch, for a total of 20 epochs. Chosen batch size was 128, with sequence padding. One consideration when creating batches was to group sentences of similar length together to minimize padding, however this would introduce bias in training procedure, and since we had sufficient hardware resources we discarded this approach. The training was conducted on NVIDIA GTX 3050 GPU, which lasted about 20-30 minutes. 

Model that produced best test loss, checkpointed model after 20 epochs, dataset and tokenizers can be found on GoogleDrive:

- Tokenizers: https://drive.google.com/drive/folders/1lClXElxyGq1VHSwTe290PTWx_ntXgZMh?usp=drive_link

- Original dataset and preprocessed data: https://drive.google.com/drive/folders/1sIMCMcVElixSHZNSfaAd3HEGwQlTixsT?usp=sharing

- Model weights: https://drive.google.com/drive/folders/1rFMguBjxm9VXVEe_O1MV5HzLvW2ohqob?usp=sharing

Overfitting can be observed at arround epoch 15, even after moderate ammount of regularization with dropout and label-smoothed cross-entropy.

## Inference
Following inference methods are implemented: greedy, topk and beam search. For topk we've implemented following hyperparameters: $k$ - width of multinomial distribution to choose next token from, **temperature** - next token probabilities are scaled by $\frac{1}{t}$. For beam search, following hyperparameters were implemented: beam_size, $\alpha \in [0, 1]$ - beam search essentially tries to maximize conditional log-likelihood, and in doing so it will naturally favor shorter sentences over longer sentences, to penalize shorter sentences log-likelihood of each candidate translation is scaled by $l^{-\alpha}$, where $l$ is the lenght of the sequence itself. $\alpha = 0$ corresponds to no scaling and $\alpha = 1$ corresponds to dividing by length itself.

After training, we've performed inference on the provided test dataset using previously described methods: greedy, topk with $k = 10$ and $t = 1$, beam search with beam size of 10 and $\alpha = 0.75$. The following results were obtained:
| Inference method | sacreBLEU score |
| ---------------- | --------------- |
| Greedy | 21.45 | 
| Beam Search | 24.86 | 
| TopK | 14.54 | 

Train and test sentences are simple in structure, and the model does produce sensible translations for them, however it fails to translate arbitrary English setnences, this is attributed to a small dataset and simple syntactic structure of sentences. There is room for improvement (choosing a larger dataset would surely be beneficial), but we emphasise that the main goal of this project was familiarizing with the Transformer architecture and it's modern variations, as well as the entire machine translation pipeline. Implemented model also supports "live inference", it can takes English sentenes from the terminal and output German translations, you can download model weights and tokenizers from the provided link if you wish to play arround with this. Do note that in this case, dataset directory and tokenizer directory will need to be adapted according to your sistem in **english_german_model.py**. Some example inferences:
```
TopK parameters: k = 10, temperature = 1
Beam search parameters: beam_size = 10, alpha = 0.75
------------------------------------------------------------------------------------------
English sentence: A man in jeans at the beach playing with a red ball.
Source sentence after preprocessing: a man in jeans at the beach playing with a red ball.

Model generated translation/s:
Greedy inference:
ein mann in jeans spielt mit einem roten ball im freien.
Inference time: 0.2450544834136963 seconds

TopK inference:
ein mann in jeans, poloshirt spielt mit dem roten ball im regen.
Inference time: 0.07501649856567383 seconds

Beam inference:
ein mann in jeans spielt am strand mit einem roten ball.
Inference time: 0.274061918258667 seconds
------------------------------------------------------------------------------------------
English sentence: Bicycle rider wearing black, riding down a dirt trail in a mountain bike.
Source sentence after preprocessing: bicycle rider wearing black, riding down a dirt trail in a mountain bike.

Model generated translation/s:
Greedy inference:
ein motorradfahrer mit schwarzen helm fährt einen feldweg in einem skatepark.
Inference time: 0.0720069408416748 seconds

TopK inference:
ein motorradfahrer mit schwarzen anzug fährt einen feldweg in einem geländemotorrad.
Inference time: 0.07501673698425293 seconds

Beam inference:
ein komplett schwarz gekleideter radfahrer fährt einen feldweg hinunter.
Inference time: 0.176039457321167 seconds
------------------------------------------------------------------------------------------
English sentence: A bald man walking down a city sidewalk while talking on his cellphone.
Source sentence after preprocessing: a bald man walking down a city sidewalk while talking on his cellphone.

Model generated translation/s:
Greedy inference:
ein glatzköpfiger mann geht auf einer straße und spricht mit seinem handy.
Inference time: 0.2759215831756592 seconds

TopK inference:
ein blonder mann in einer bibliothek geht einen gehweg hinunter, während er auf seiner gitarre.
Inference time: 0.12800335884094238 seconds

Beam inference:
ein kahlköpfiger mann geht eine straße entlang und telefoniert mit seinem handy.
Inference time: 0.08301830291748047 seconds
```