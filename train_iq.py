import argparse
import os
import pickle
from typing import OrderedDict

from torch._C import device
from utils.vocab import build_vocab, load_vocab
from utils.data_loader import get_loader
from utils import NLGEval
from numpy.lib.function_base import gradient
from torchvision.transforms import transforms
from copy import deepcopy
from models import IQ
import math
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class TrainIQ(pl.LightningModule):
    def __init__(self, vocab, args):
        super().__init__()
        
        self.latent_transformer = False
        self.vocab = vocab
        self.args = args

        self.iter = 0
        self.kliter = 0
        self.nlge = NLGEval(no_glove=True, no_skipthoughts=True)
        metrics = {
            "loss": [],
            "img": [],
            "ppl": [],
            "kld": [],
            "aux": [],
            "elbo": [],
            "rec": [],
        }
        self.val_metrics = deepcopy(metrics)

        self.model = IQ(self.latent_transformer, vocab, args)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word2idx[self.vocab.SYM_PAD])
        self.image_recon_criterion = nn.MSELoss()

    def token_decode(self, tokenized_tensor_of_ints, sample=5):
        for i, batch_item in enumerate(tokenized_tensor_of_ints):
            if i == sample:
                break
            sentence_string = " ".join([self.vocab.idx2word[token.item()] for token in batch_item])
            print(sentence_string)
        print()

    def forward(self, batch):
        images, _, questions, posteriors, answers, _, _ = batch.values()
        images, questions, posteriors, answers = images.cuda(), questions.cuda(), posteriors.cuda(), answers.cuda()
        output, z, kld_loss, image_recon = self.model(images, answers, posteriors, questions)
        return output, z, kld_loss, image_recon

    def calculate_losses(self, output, image_recon, kld_loss, z_logit, target):        
        loss_rec = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
        loss_img = self.image_recon_criterion(image_recon[0], image_recon[1])

        if not self.latent_transformer:
            loss = loss_rec + self.args.image_recon_lambda * loss_img
            elbo = loss_rec
            kld_loss = torch.Tensor([0])
            aux = 0
        else:
            z_logit = z_logit.unsqueeze(1).repeat(1, output.size(1), 1)
            loss_aux = self.criterion(z_logit.reshape(-1, z_logit.size(-1)), target.reshape(-1))

            kl_weight = min(math.tanh(6 * self.kliter/self.args.full_kl_step - 3) + 1, 1)
            aux = loss_aux.item()
            elbo = loss_rec + kld_loss
            loss = loss_rec + self.args.kl_ceiling * kl_weight * kld_loss + self.args.aux_ceiling*loss_aux + self.args.image_recon_lambda * loss_img

        return loss, loss_rec.item(), loss_img.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item(), aux, elbo.item()

    def training_step(self, batch, batch_idx):

        # switch to latent transformer if we've reached num_pretraining_steps
        if self.iter == self.args.num_pretraining_steps:
            self.latent_transformer = True
            self.model.switch_GVT_train_mode(self.latent_transformer)

        output, z_logit, kld_loss, image_recon = self(batch)
        target = batch["questions"].cuda()

        loss, loss_rec, loss_img, ppl, kld_loss, aux, elbo = self.calculate_losses(output, image_recon, kld_loss, z_logit, target)

        if self.latent_transformer: self.kliter += 1

        self.log('train loss', loss)
        self.log('train rec loss', loss_rec)
        self.log('image recon loss', loss_img)
        self.log('perplexity', ppl)
        self.log('kld loss', kld_loss)
        self.log('aux loss', aux)
        self.log('elbo', elbo)

        self.iter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch["questions"].cuda()
        output, z_logit, kld_loss, image_recon = self(batch)

        loss, loss_rec, loss_img, ppl, kld_loss, aux, elbo = self.calculate_losses(output, image_recon, kld_loss, z_logit, target)

        self.val_metrics["loss"].append(loss.item())
        self.val_metrics["img"].append(self.args.image_recon_lambda * loss_img)
        self.val_metrics["ppl"].append(ppl)
        self.val_metrics["kld"].append(kld_loss)
        self.val_metrics["aux"].append(aux)
        self.val_metrics["elbo"].append(elbo)
        self.val_metrics["rec"].append(loss_rec)

        self.log("val_loss", loss.item())
        self.log("val_loss_rec", loss_rec)
        self.log("val_img_loss", loss_img)
        self.log("val_ppl", ppl)
        self.log("val_kld_loss", kld_loss)
        self.log("val_aux", aux)
        self.log("val_elbo", elbo)

        return batch

    def validation_epoch_end(self, batch) -> None:
        
        print("##### End of Epoch validation #####")

        batch = batch[0]
        
        categories = batch["answer_types"].cuda().unsqueeze(-1)
        images = batch["images"].cuda()
        image_ids = batch["image_ids"]


        ref, hyp_g = [], []
        decoded_sentences = self.model.decode_greedy(images, categories, max_decode_length=50)
        print("VALIDATION SAMPLE")
        for i, greedy_sent in enumerate(decoded_sentences):
            if i == 10: break
            rf = " ".join([self.vocab.idx2word[word] for word in batch["questions"][i].tolist()])
            hyp_g.append(greedy_sent)
            ref.append(rf)
            # print("Emotion:\t", batch["program_txt"][i])
            print("Image ID:\t", image_ids[i])
            print("Context:\t", " ".join([self.vocab.idx2word[category] for category in categories[i].tolist()]))
            print("Generated: \t", greedy_sent)
            print("Reference: \t", rf)
            print()

        for k, v in self.val_metrics.items():
            print(k, "\t", np.round(np.mean(v), 4))
            self.val_metrics[k] = [] # reset v
        print()


    def filter_special_tokens(self, decoded_sentence_list):
        filtered = []
        special_tokens = ["<start>", "<end>", "<pad>"]
        for token in decoded_sentence_list:
            if token not in special_tokens:
                filtered.append(token)
        return filtered


    def test_step(self, batch, batch_idx):
        images, questions, answers, categories = batch["images"], batch["questions"], batch["answers"], batch["answer_types"]
        images, questions, answers, categories = images.to(self.args.device), questions.to(self.args.device), answers.to(self.args.device), categories.to(self.args.device)
        categories = categories.unsqueeze(1)
        
        preds = []
        gts = []
        decoded_sentences = self.model.decode_greedy(images, categories, max_decode_length=50)
        for i, greedy_sentence in enumerate(decoded_sentences):
            list_gt = self.filter_special_tokens([self.vocab.idx2word[word] for word in batch["questions"][i].tolist()])
            list_pred = self.filter_special_tokens(greedy_sentence.split())
            gt = " ".join(list_gt)
            pred = " ".join(list_pred)
            gts.append(gt)
            preds.append(pred)

        scores = self.nlge.compute_metrics(ref_list=[gts], hyp_list=preds)

        for k, v in scores.items():
            scores[k] = torch.tensor(v)

        return scores


    def test_end(self, all_scores):
        for k, scores in all_scores.items():
            all_scores[k] = scores.detach().cpu().numpy()
            all_scores[k] = np.mean(all_scores[k])

        print(all_scores)
        return all_scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224,
                                    scale=(1.00, 1.2),
                                    ratio=(0.75, 1.3333333333333333)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

class MyEarlyStopping(EarlyStopping):
    def __init__(self, monitor: str, min_delta: float, patience: int, verbose: bool, mode: str, strict: bool, early_stop_from):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, strict=strict)
        self.early_stop_from = early_stop_from

    def on_validation_end(self, trainer, pl_module):
        if pl_module.iter > self.early_stop_from:
            self._run_early_stopping_check(trainer, pl_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Model hyperparameters
    parser.add_argument("--emb-dim", type=int, default=300, help="Embedding dimensionality of the model")
    parser.add_argument("--hidden-dim", type=int, default=300, help="Hidden dimensionality of the model")
    parser.add_argument("--latent_dim", type=int, default=300, help="Size of latent dimension")
    parser.add_argument("--pwffn-dim", type=int, default=600, help="Size of postionwise feedforward network in transformer")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers in encoder and decoder")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads in the multi-head attention")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of the network")
    parser.add_argument("--num_pretraining-steps", type=float, default=8000, help="Number of pretraining steps before turning on latent transformer")
    parser.add_argument("--total-training-steps", type=int, default=30000, help="Total number of training steps for the model")
    parser.add_argument("--full_kl_step", type=int, default=9000, help="Number of steps until KLD is annealed")
    parser.add_argument("--kl_ceiling", type=float, default=0.08)
    parser.add_argument("--aux_ceiling", type=float, default=1.0)
    parser.add_argument("--image-recon-lambda", type=float, default=0.1, help="How much to scale the image reconstruction loss by")
    parser.add_argument("--batch-size", type=int, default=128)
    ## Data args
    parser.add_argument("--emb-file", type=str, default="vectors/glove.6B.300d.txt", help="Filepath for pretrained embeddings")
    parser.add_argument("--dataset", type=str, default="data/processed/iq_dataset.hdf5")
    parser.add_argument("--val-dataset", type=str, default="data/processed/iq_val_dataset.hdf5")
    parser.add_argument("--vocab", type=str, default="vocab.pkl")
    parser.add_argument("--use-gpu", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    args.device = device

    early_stop_callback = MyEarlyStopping(
        monitor='train loss',
        min_delta=0.00,
        patience=8,
        verbose=True,
        mode='min',
        early_stop_from=args.num_pretraining_steps
    )

    data_loader = get_loader(args.dataset, transform, 128, shuffle=False, num_workers=8)
    val_data_loader = get_loader(args.val_dataset, transform, 128, shuffle=True, num_workers=8)
    if os.path.exists(args.vocab):
        vocab = pickle.load(open(args.vocab, "rb"))
    else: 
        vocab = build_vocab('data/vqa/v2_OpenEnded_mscoco_train2014_questions.json', 'data/vqa/iq_dataset.json', 4)
    trainGVT = TrainIQ(vocab, args).to(args.device)
    trainer = pl.Trainer(max_steps=args.total_training_steps, gradient_clip_val=5, val_check_interval=200, callbacks=[early_stop_callback], limit_val_batches=0.1)
    trainer.fit(trainGVT, data_loader, val_data_loader)
    trainer.test(trainGVT, test_dataloaders=val_data_loader)
