"""CaReFlow model: map acoustic/visual distributions onto the language
distribution with cyclic adaptive rectified flow, then fuse for prediction.

Implementation of:
    Eq. (3)/(6)   forward Euler integration at inference,
    Eq. (7)/(8)   adaptive relaxed forward-flow loss,
    Eq. (11)      cyclic backward-flow loss,
    Eq. (12)      joint optimization with the main task loss.
"""

from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
)
from transformers.models.bert.modeling_bert import BertPooler
import global_configs
from global_configs import DEVICE
from modules.transformer import TransformerEncoder
import torch
import numpy as np

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

from rectified_flow_new import RectifiedFlow
from model_new import Prediction

# Default text backbone. Can be overridden through multimodal_config.text_model.
DEFAULT_TEXT_MODEL = "microsoft/deberta-v3-base"


class DebertaModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM,
            global_configs.ACOUSTIC_DIM,
            global_configs.VISUAL_DIM,
        )
        self.config = config
        # Kept for state-dict compatibility with the released checkpoint; it
        # is not used in the forward pass.
        self.pooler = BertPooler(config)
        text_model_name = getattr(multimodal_config, "text_model", None) or DEFAULT_TEXT_MODEL
        self.model = DebertaV2Model.from_pretrained(text_model_name).to(DEVICE)

        self.ratio = multimodal_config.ratio
        self.d_l = multimodal_config.share_dim
        self.attn_dropout = multimodal_config.drop_prob
        self.unimodal_dropout = multimodal_config.dropout_unimodal
        self.unimodal_head = multimodal_config.transformer_head
        self.label_dim = 1

        self.proj_a = nn.Conv1d(
            ACOUSTIC_DIM, self.d_l,
            kernel_size=multimodal_config.kernel_size, stride=1,
            padding=(multimodal_config.kernel_size - 1) // 2, bias=False,
        )
        self.proj_l = nn.Linear(TEXT_DIM, self.d_l, bias=False)
        self.proj_v = nn.Conv1d(
            VISUAL_DIM, self.d_l,
            kernel_size=multimodal_config.kernel_size, stride=1,
            padding=(multimodal_config.kernel_size - 1) // 2, bias=False,
        )
        self.transa = self.get_network(self_type="a", layers=multimodal_config.transformer_layer)
        self.transv = self.get_network(self_type="v", layers=multimodal_config.transformer_layer)

        # Forward velocity fields V_{a,l}, V_{v,l} and backward fields.
        self.reflow_a = Prediction(base_channels=self.d_l)
        self.reflow_v = Prediction(base_channels=self.d_l)
        self.reflow_a_b = Prediction(base_channels=self.d_l)
        self.reflow_v_b = Prediction(base_channels=self.d_l)
        self.rf_a = RectifiedFlow()
        self.rf_v = RectifiedFlow()
        self.rf_a_b = RectifiedFlow()
        self.rf_v_b = RectifiedFlow()

        self.LayerNorm_l = nn.LayerNorm(self.d_l)
        self.LayerNorm_a = nn.LayerNorm(self.d_l)
        self.LayerNorm_v = nn.LayerNorm(self.d_l)

        self.fusion = nn.Sequential(
            nn.Linear(self.d_l * 3, multimodal_config.inter_dim),
            nn.ReLU(),
            nn.Linear(multimodal_config.inter_dim, self.d_l),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.d_l, multimodal_config.inter_dim),
            nn.ReLU(),
            nn.Linear(multimodal_config.inter_dim, 1),
        )

        self.step_size = multimodal_config.step_size
        self.eps = multimodal_config.eps
        # Opt-in correct masking; default False reproduces the paper checkpoint.
        self.use_attention_mask = getattr(multimodal_config, "use_attention_mask", False)

        # self.init_weights()  # weights are loaded by the outer from_pretrained.

    def get_network(self, self_type="l", layers=3):
        if self_type in ["l", "a", "v"]:
            embed_dim, attn_dropout, transformer_dropout, transformer_head = (
                self.d_l, self.attn_dropout, self.unimodal_dropout, self.unimodal_head
            )
        elif self_type == "lav":
            embed_dim, attn_dropout, transformer_dropout, transformer_head = (
                self.d_l * 3, self.attn_dropout, self.unimodal_dropout, self.unimodal_head
            )
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=transformer_head,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=transformer_dropout,
            res_dropout=transformer_dropout,
            embed_dropout=transformer_dropout,
            attn_mask=False,
        )

    # ------------------------------------------------------------------ #
    # Rectified-flow helpers
    # ------------------------------------------------------------------ #
    def _euler_integrate(self, velocity_net, rf, x_source):
        """Integrate from a source feature to the language distribution (Eq. 6).

        Starts at t = 0 (the source) and takes ``step_size`` Euler steps of
        size dt = 1 / step_size. ``x_t.detach()`` prevents the main-task loss
        from back-propagating through successive solver steps; each velocity
        prediction still receives gradients w.r.t. the velocity parameters.
        """
        dt = 1.0 / self.step_size
        x_t = x_source
        for j in range(self.step_size):
            t = torch.tensor([j * dt], device=x_source.device)
            v_pred = velocity_net(x=x_t.detach(), t=t)
            x_t = rf.euler(x_t, v_pred, dt)
        return x_t

    def _forward_flow_loss(self, velocity_net, rf, output_source, output_l,
                           label_ids, same_eta):
        """One-to-many adaptive-relaxed forward flow loss (Eqs. 7-8).

        Pairs are (target=language, source=acoustic/visual):
          * first B pairs are same-sample pairs (i, i) with margin eta = 0;
          * the remaining pairs are cross-sample pairs (index1, index2) with
            eta = eps + (y_i - y_j)^2; self-pairs are filtered out.
        Returns:
            loss, and a one-step estimate of the transferred feature used by
            the backward flow (built from the same-sample velocity only).
        """
        bsz = output_l.shape[0]
        candidates = list(range(bsz))
        index1 = np.random.choice(candidates, bsz * self.ratio, replace=True)
        index2 = np.random.choice(candidates, bsz * self.ratio, replace=True)
        keep = np.where(index1 != index2)[0]
        index1, index2 = index1[keep], index2[keep]

        x_1 = torch.cat([output_l, output_l[index1, :]]).detach()  # target
        x_0 = torch.cat([output_source, output_source[index2, :]])#.detach()  # source

        cross_label_dist = (
            label_ids.view(-1)[index1] - label_ids.view(-1)[index2]
        ) ** 2
        eta = torch.cat(
            [same_eta, cross_label_dist.to(same_eta.device) + self.eps], dim=0
        )

        t = torch.rand(x_1.size(0), device=output_l.device)
        x_t, _ = rf.create_flow(x_1, t, x_0)
        v_pred = velocity_net(x=x_t, t=t)

        # One-step transferred feature for the cyclic loss, from the B
        # same-sample predictions (accurate one-to-one correspondence).
        trans_feat = output_source.detach() + v_pred[:bsz, :]
        loss = rf.mse_loss(v_pred, x_1, x_0, eta)
        return loss, trans_feat

    @staticmethod
    def _backward_flow_loss(velocity_net_b, rf_b, output_source, trans_feat):
        """Cyclic backward flow loss (Eq. 11).

        target = original source feature X_{m1} (detached);
        source = forward output X_{m1,m2}, which is intentionally NOT detached
        so the backward loss can shape the forward velocity field.
        """
        x_1 = output_source.detach()
        x_0 = trans_feat
        t = torch.rand(x_1.size(0), device=x_1.device)
        x_t, _ = rf_b.create_flow(x_1, t, x_0)
        v_pred = velocity_net_b(x=x_t, t=t)
        return rf_b.mse_loss(v_pred, x_1, x_0)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def _pool_text(self, hidden, input_mask):
        if self.use_attention_mask and input_mask is not None:
            m = input_mask.unsqueeze(-1).to(hidden.dtype)
            return (self.LayerNorm_l(hidden) * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.LayerNorm_l(hidden).mean(dim=1)

    def forward(self, input_ids, visual, acoustic, label_ids=None,
                input_mask=None):
        # ---------------- Unimodal encoders ---------------- #
        attn_mask = input_mask if self.use_attention_mask else None
        x = self.model(input_ids, attention_mask=attn_mask)[0]
        output_l = self.proj_l(x)

        acoustic = self.proj_a(acoustic.transpose(1, 2)).permute(2, 0, 1)  # (t,b,d)
        visual = self.proj_v(visual.transpose(1, 2)).permute(2, 0, 1)
        output_a = self.transa(acoustic).permute(1, 0, 2)                 # (b,t,d)
        output_v = self.transv(visual).permute(1, 0, 2)

        output_l = self._pool_text(output_l, input_mask)
        output_a = self.LayerNorm_a(output_a).mean(dim=1)
        output_v = self.LayerNorm_v(output_v).mean(dim=1)

        # ---------------- Inference-time distribution mapping (Eq. 6) ---- #
        output_a_trans = self._euler_integrate(self.reflow_a, self.rf_a, output_a)
        output_v_trans = self._euler_integrate(self.reflow_v, self.rf_v, output_v)

        fusion = self.fusion(
            torch.cat([output_l, output_a_trans, output_v_trans], dim=-1)
        )
        pooled_output = self.predictor(fusion)

        # ---------------- Auxiliary cyclic adaptive flow losses ---------- #
        loss_f, loss_b = 0.0, 0.0
        if self.training:
            if label_ids is None:
                raise ValueError("label_ids are required to form the flow losses.")
            bsz = output_l.shape[0]
            same_eta = torch.zeros(bsz, device=output_l.device)

            loss_fa, output_a_trans = self._forward_flow_loss(
                self.reflow_a, self.rf_a, output_a, output_l, label_ids, same_eta
            )
            loss_fv, output_v_trans = self._forward_flow_loss(
                self.reflow_v, self.rf_v, output_v, output_l, label_ids, same_eta
            )
            loss_f = loss_fa + loss_fv

            loss_b = self._backward_flow_loss(
                self.reflow_a_b, self.rf_a_b, output_a, output_a_trans
            )
            loss_b += self._backward_flow_loss(
                self.reflow_v_b, self.rf_v_b, output_v, output_v_trans
            )

        return pooled_output, loss_f, loss_b


class DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dberta = DebertaModel(config, multimodal_config)
        self.init_weights()

    def forward(self, input_ids, visual, acoustic, label_ids=None,
                input_mask=None):
        return self.dberta(
            input_ids, visual, acoustic, label_ids=label_ids,
            input_mask=input_mask,
        )
