# CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion

Official implementation of **CaReFlow** (CVPR 2026). CaReFlow maps the
distributions of the acoustic and visual modalities onto the language
distribution with a *cyclic, adaptive* rectified flow before multimodal
fusion, narrowing the modality gap for multimodal affective computing.

```
source (a/v)  --forward V_{m,l}-->  language-aligned feature --fusion--> prediction
                     ^---backward V_hat_{m,l}--- (cyclic, Eq. 11)
```

## Repository layout

```
.
├── model.py                 # velocity field MLP V_{m1,m2} (Eqs. 9-10)
├── rectified_flow.py        # interpolation / Euler / adaptive-relaxed loss (Eqs. 1,2,7)
├── model_reflow_clean.py    # full CaReFlow model (forward + cyclic backward flows)
├── train_reflow_clean.py    # training / evaluation / testing entry point
├── global_configs.py        #  see global_configs.py
├── modules/
│   └── transformer.py       # TransformerEncoder (MulT-style)
├── datasets/
│   ├── mosi.pkl             #
│   └── mosei.pkl            #
└── microsoft/
    ├── deberta-v3-base      #
        ...                  # 
```



## Setup

```bash
pip install -r requirements.txt
```

`DebertaV2Tokenizer` needs `sentencepiece` (already in requirements). The
first run downloads `microsoft/deberta-v3-base`.

You can download DeBERTa-v3-base from huggingface (https://huggingface.co/microsoft/deberta-v3-base).


## Data format

`datasets/{dataset}.pkl` is a dict `{"train": ..., "dev": ..., "test": ...}`.
Each example is `((words, visual, acoustic), label, segment)`, where `words`
is a list of word strings and `visual` / `acoustic` are `(T, d_m)` arrays
aligned word-by-word (the standard CMU-MOSI/MOSEI format).

Download the datasets to `./datasets` by running `download_datasets.sh`.

## Run

```bash
# CMU-MOSI
python train_reflow_new.py --dataset mosi --n_epochs 100 --train_batch_size 32 --learning_rate 10e-6 --ratio 4 --dropout_prob 0.5 --inter_dim 150 --share_dim 100 --transformer_layer 3 --step_size 2 --loss_b_ratio 0.1 --loss_f_ratio 0.2 --eps 1e-3

# CMU-MOSEI
CUDA_VISIBLE_DEVICES=0 python train_reflow_new.py --dataset mosei --n_epochs 10 --train_batch_size 50 --learning_rate 10e-6 --ratio 5 --dropout_prob 0.5 --inter_dim 150 --share_dim 150 --transformer_layer 3 --step_size 2 --loss_b_ratio 0.2 --loss_f_ratio 0.1 --eps 1e-5 
```

Key arguments and their paper symbols:

| Argument | Symbol | Meaning | Default |
|---|---|---|---|
| `--ratio` | β | cross-sample pairs × same-sample pairs | 4 |
| `--step_size` | 1/dt | Euler steps for inference integration | 3 |
| `--loss_f_ratio` | α_f | forward-flow loss weight | 0.005 |
| `--loss_b_ratio` | α_b | backward-flow loss weight | 0.005 |
| `--eps` | ε | margin floor η for cross-sample pairs | 1e-5 |
| `--save_model` | – | save best-by-validation checkpoint | off |
| `--use_attention_mask` | – | mask padding in the text encoder & pooling | off |


## Acknowledgement

We thank the authors of "https://github.com/joshuaxiao98/ITHP", "https://github.com/WasifurRahman/BERT_multimodal_transformer", and "https://github.com/TongTong313/rectified-flow" for sharing their codes.


## Citation

```bibtex
@InProceedings{Mai_2026_CVPR,
    author    = {Mai, Sijie and Han, Shiqin},
    title     = {CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {37799-37809}
}
```

