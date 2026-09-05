import argparse
import os
import random
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.nn import MSELoss
from transformers import (
    get_linear_schedule_with_warmup,
    DebertaV2Tokenizer,
)
from torch.optim import AdamW
from model_reflow_new import DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=50)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=128)
parser.add_argument("--inter_dim", default=100, type=int)
parser.add_argument("--drop_prob", default=0.3, type=float)
parser.add_argument("--beta_shift", default=1.0, type=float)
parser.add_argument("--share_dim", default=100, type=int)
parser.add_argument("--pretrained_epoch", default=30, type=int)
parser.add_argument("--ratio", default=4, type=int, help="one-to-many ratio beta")
parser.add_argument("--loss_f_ratio", default=0.005, type=float, help="alpha_f")
parser.add_argument("--loss_b_ratio", default=0.005, type=float, help="alpha_b")
parser.add_argument("--dropout_unimodal", default=0.3, type=float)
parser.add_argument("--transformer_head", default=5, type=int)
parser.add_argument("--transformer_layer", default=3, type=int)
parser.add_argument("--kernel_size", default=3, type=int)
parser.add_argument("--pretrain_iterations", default=300, type=int)
parser.add_argument("--step_size", default=2, type=int, help="number of Euler steps")
parser.add_argument("--threshold", default=0.6, type=float)
parser.add_argument("--margin", default=0.1, type=float)
parser.add_argument("--eps", default=1e-5, type=float, help="epsilon in eta (Eq. 8)")
# ---- release-quality options (defaults reproduce the paper) ----
parser.add_argument("--output_dir", type=str, default="checkpoints")
parser.add_argument("--save_model", action="store_true",
                    help="save the best-by-validation checkpoint")
parser.add_argument("--use_attention_mask", action="store_true",
                    help="use attention mask in the text encoder and masked pooling")
args = parser.parse_args()
# Propagate opt-in options to the model (which receives this Namespace as
# multimodal_config). text_model mirrors --model so the backbone is not
# hard-coded inside the model.
args.text_model = args.model
args.use_attention_mask = args.use_attention_mask

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (
    global_configs.ACOUSTIC_DIM,
    global_configs.VISUAL_DIM,
    global_configs.TEXT_DIM,
)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids,
                 label_id, sample_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sample_id = sample_id


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))
        assert len(tokens) == len(inversions)

        aligned_visual, aligned_audio = [], []
        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])
        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_deberta_input(
            tokens, visual, acoustic, tokenizer
        )

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
                sample_id=segment,
            )
        )
    return features


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    pad_length = args.max_seq_length - len(input_ids)

    acoustic = np.concatenate((acoustic, np.zeros((pad_length, ACOUSTIC_DIM))))
    visual = np.concatenate((visual, np.zeros((pad_length, VISUAL_DIM))))
    padding = [0] * pad_length
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


# Shared sample-id bookkeeping across train/dev/test splits.
sample_dict, sample_dict2 = {}, {}


def get_appropriate_dataset(data):
    tokenizer = get_tokenizer(args.model)
    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)
    '''
    for f in features:
        if f.sample_id not in sample_dict2:
            new_id = len(sample_dict)
            sample_dict2[f.sample_id] = new_id
            sample_dict[new_id] = f.sample_id
    all_sample_ids = torch.tensor(
        np.array([sample_dict2[f.sample_id] for f in features]), dtype=torch.float
    )
    '''
   # all_sample_ids = {}
    return TensorDataset(
        all_input_ids, all_visual, all_acoustic, all_label_ids, all_input_mask,
    )


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)
    train_data, dev_data, test_data = data["train"], data["dev"], data["test"]
    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step) * args.n_epochs
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False
    )
    return (train_dataloader, dev_dataloader, test_dataloader,
            num_train_optimization_steps)


def set_random_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed: {}".format(seed))


def prep_for_training(num_train_optimization_steps: int):
    model = DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )
    model.to(DEVICE)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def batch_minmax(x):
    """Per-batch min-max normalization (matches the released pipeline).

    Caveat: statistics are computed per batch, so train/eval use different
    batch sizes. Kept for exact reproducibility; +1e-8 only guards against
    division by zero on a constant batch.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def train_epoch(model, train_dataloader, optimizer, scheduler):
    model.train()
    tr_loss, nb_tr_steps = 0.0, 0
    total_loss_f, total_loss_b = [], []
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids, input_mask = batch
        visual = batch_minmax(torch.squeeze(visual, 1))
        acoustic = batch_minmax(torch.squeeze(acoustic, 1))

        logits, loss_f, loss_b = model(
            input_ids, visual, acoustic, label_ids, input_mask=input_mask
        )
        main_loss = MSELoss()(logits.view(-1), label_ids.view(-1))
        loss = (main_loss
                + args.loss_f_ratio * loss_f
                + args.loss_b_ratio * loss_b)  # Eq. (12)

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()
        tr_loss += loss.item()
        nb_tr_steps += 1
        total_loss_f.append(loss_f.item() if torch.is_tensor(loss_f) else loss_f)
        total_loss_b.append(loss_b.item() if torch.is_tensor(loss_b) else loss_b)

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return tr_loss / nb_tr_steps, total_loss_f, total_loss_b


def _forward_eval(model, batch):
    input_ids, visual, acoustic, label_ids, input_mask = batch
    visual = batch_minmax(torch.squeeze(visual, 1))
    acoustic = batch_minmax(torch.squeeze(acoustic, 1))
    return model(input_ids, visual, acoustic, label_ids, input_mask=input_mask)


def eval_epoch(model, dev_dataloader):
    model.eval()
    dev_loss, nb_dev_steps = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Iteration"):
            batch = tuple(t.to(DEVICE) for t in batch)
            label_ids = batch[3]
            logits, _, _ = _forward_eval(model, batch)
            loss = MSELoss()(logits.view(-1), label_ids.view(-1))
            dev_loss += loss.item()
            nb_dev_steps += 1
    return dev_loss / nb_dev_steps


def test_epoch(model, test_dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            label_ids = batch[3]
            logits, _, _ = _forward_eval(model, batch)
            preds.extend(np.squeeze(logits.detach().cpu().numpy()).tolist())
            labels.extend(np.squeeze(label_ids.detach().cpu().numpy()).tolist())
    return np.array(preds), np.array(labels)


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def test_score_model(model, test_dataloader, use_zero=False):
    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])
    mult_a7 = multiclass_acc(
        np.clip(preds, -3.0, 3.0), np.clip(y_test, -3.0, 3.0)
    )
    mae_non = np.mean(np.absolute(preds - y_test))
    corr_non = np.corrcoef(preds, y_test)[0][1]

    preds, y_test = preds[non_zeros], y_test[non_zeros]
    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    preds, y_test = preds >= 0, y_test >= 0
    return (accuracy_score(y_test, preds), mae, corr,
            f1_score(y_test, preds, average="weighted"),
            mult_a7, mae_non, corr_non)


def train(model, train_dataloader, validation_dataloader, test_data_loader,
          optimizer, scheduler):
    best_valid_loss = float("inf")
    best_metrics = None
    last_metrics = None
    for epoch_i in range(int(args.n_epochs)):
        train_loss, _, _ = train_epoch(
            model, train_dataloader, optimizer, scheduler
        )
        valid_loss = eval_epoch(model, validation_dataloader)
        print("TRAIN: epoch:{}, train_loss:{}, valid_loss:{}".format(
            epoch_i + 1, train_loss, valid_loss))

        (test_acc, test_mae, test_corr, test_f_score, mult_a7,
         test_mae_non, test_corr_non) = test_score_model(model, test_data_loader)
        last_metrics = (test_acc, test_mae, test_corr, test_f_score, mult_a7)
        print(
            "TEST: train_loss:{}, valid_loss:{}, test_acc:{}, mae:{}, corr:{}, "
            "f1_score:{}, mult_a7:{}, mae_non:{}, corr_non:{}".format(
                train_loss, valid_loss, test_acc, test_mae, test_corr,
                test_f_score, mult_a7, test_mae_non, test_corr_non))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_metrics = (test_acc, test_mae_non, test_corr_non,
                            test_f_score, mult_a7)
            if args.save_model:
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt = os.path.join(
                    args.output_dir, f"careflow_{args.dataset}_best.pt")
                torch.save(model.state_dict(), ckpt)
                print("Saved best checkpoint to", ckpt)

        print("BEST TEST: best_acc:{}, best_mae_non:{}, best_corr_non:{}, "
              "best_f1_score:{}, best_acc7:{}".format(*best_metrics))
    return best_valid_loss, best_metrics, last_metrics


def main():
    set_random_seed(args.seed)
    (train_data_loader, dev_data_loader, test_data_loader,
     num_train_optimization_steps) = set_up_data_loader()
    model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)
    train(model, train_data_loader, dev_data_loader, test_data_loader,
          optimizer, scheduler)


if __name__ == "__main__":
    main()
