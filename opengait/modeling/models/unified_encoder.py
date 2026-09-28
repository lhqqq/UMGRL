import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def resample_cloud_tokens(z_cloud: torch.Tensor, target_parts: int = 16) -> torch.Tensor:
    return F.interpolate(
        z_cloud,
        size=target_parts,
        mode='linear',
        align_corners=False
    )


class SharedEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 num_parts: int = 16, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_parts = num_parts

        self.part_embed = nn.Parameter(torch.randn(1, num_parts, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, P = z.shape

        x = z.permute(0, 2, 1)
        x = x + self.part_embed[:, :P, :]
        x = self.transformer(x)
        x = self.out_proj(x)
        z_out = x.permute(0, 2, 1)

        return z_out


class Decomposition(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()

        self.d_model = d_model

        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.private_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.shared_head, self.private_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, P = z.shape

        x = z.permute(0, 2, 1)

        c = self.shared_head(x)
        r = self.private_head(x)

        c = c.permute(0, 2, 1)
        r = r.permute(0, 2, 1)

        return c, r


class GatedRecomposition(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        c_pooled = c.mean(dim=-1)

        alpha = self.gate_mlp(c_pooled)

        alpha = alpha.unsqueeze(-1)

        y = c + alpha * r

        return y


class ModalityImputer(nn.Module):
    def __init__(self, d_model: int = 256, num_parts: int = 16,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()

        self.d_model = d_model
        self.num_parts = num_parts
        self.concat_dim = d_model * 2

        self.part_queries = nn.ParameterDict({
            'seg': nn.Parameter(torch.randn(num_parts, d_model) * 0.02),
            'pose': nn.Parameter(torch.randn(num_parts, d_model) * 0.02),
            'cloud': nn.Parameter(torch.randn(num_parts, d_model) * 0.02),
        })

        self.input_proj = nn.Linear(self.concat_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.out_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, visible_tokens: torch.Tensor,
                missing_modalities: List[str],
                key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = visible_tokens.size(0)

        memory = self.input_proj(visible_tokens)

        y_hat = {}
        for modal in missing_modalities:
            if modal in self.part_queries:
                queries = self.part_queries[modal].unsqueeze(0).expand(B, -1, -1)

                decoded = self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)

                out = self.out_proj(decoded)

                y_hat[modal] = out.permute(0, 2, 1)

        return y_hat

    def prepare_visible_tokens(self, decomposed: Dict,
                               visible_mask: torch.Tensor,
                               modality_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        B = visible_mask.size(0)
        token_list = []
        mask_list = []

        for m_idx, modal in enumerate(modality_names):
            if modal in decomposed and isinstance(decomposed[modal], dict):
                c = decomposed[modal]['c']
                r = decomposed[modal]['r']
                cr = torch.cat([c, r], dim=1)
                cr = cr.permute(0, 2, 1)
            else:
                ref = None
                for m_name in modality_names:
                    if m_name in decomposed and isinstance(decomposed[m_name], dict):
                        ref = decomposed[m_name]
                        break
                if ref is None:
                    raise ValueError("No modality embeddings available for token padding")
                c_ref = ref['c']
                _, C, P = c_ref.shape
                cr = torch.zeros(B, P, C * 2, device=c_ref.device, dtype=c_ref.dtype)

            token_list.append(cr)

            modal_pad = ~visible_mask[:, m_idx].unsqueeze(1).expand(B, cr.size(1))
            mask_list.append(modal_pad)

        tokens = torch.cat(token_list, dim=1)
        key_padding_mask = torch.cat(mask_list, dim=1)
        return tokens, key_padding_mask


class ReliabilityHead(nn.Module):
    def __init__(self, d_model: int = 256, tau: float = 0.1):
        super().__init__()

        self.tau = tau
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.mlp(h)
        p_pred = F.softmax(logits / self.tau, dim=-1)
        w = p_pred[:, 0]
        return p_pred, w


def dropout_sampler(available_modalities: List[str],
                    drop_probs: Tuple[float, float, float] = (0.2, 0.7, 0.1),
                    two_modal_drop_probs: Tuple[float, float] = (0.3, 0.7),
                    generator: Optional[torch.Generator] = None
                    ) -> Tuple[List[str], List[str]]:
    n = len(available_modalities)

    if n == 1:
        return available_modalities.copy(), []

    if n == 2:
        keep_both, drop_one = two_modal_drop_probs
        if abs(keep_both + drop_one - 1.0) > 1e-6:
            raise ValueError('two_modal_drop_probs must sum to 1.')
        r = torch.rand(1, generator=generator).item()
        if r < keep_both:
            return available_modalities.copy(), []
        else:
            drop_idx = int(torch.randint(0, 2, (1,), generator=generator).item())
            visible = [m for i, m in enumerate(available_modalities) if i != drop_idx]
            missing = [available_modalities[drop_idx]]
            return visible, missing

    r = torch.rand(1, generator=generator).item()
    keep_all, drop_1, drop_2 = drop_probs
    if abs(keep_all + drop_1 + drop_2 - 1.0) > 1e-6:
        raise ValueError('drop_probs must sum to 1.')

    if r < keep_all:
        return available_modalities.copy(), []
    elif r < keep_all + drop_1:
        drop_idx = int(torch.randint(0, 3, (1,), generator=generator).item())
        visible = [m for i, m in enumerate(available_modalities) if i != drop_idx]
        missing = [available_modalities[drop_idx]]
        return visible, missing
    else:
        keep_idx = int(torch.randint(0, 3, (1,), generator=generator).item())
        visible = [available_modalities[keep_idx]]
        missing = [m for i, m in enumerate(available_modalities) if i != keep_idx]
        return visible, missing


class UnifiedEncoderV2(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 num_parts: int = 16, dropout: float = 0.1, de_tau: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_parts = num_parts

        self.encoder = SharedEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            num_parts=num_parts,
            dropout=dropout
        )

        self.decomposition = Decomposition(d_model=d_model)

        self.recomposition = GatedRecomposition(d_model=d_model)

        self.reliability_head = ReliabilityHead(d_model=d_model, tau=de_tau)

        self.imputer = ModalityImputer(
            d_model=d_model,
            num_parts=num_parts,
            n_heads=n_heads,
            n_layers=2
        )

        self.modality_names = ['seg', 'pose', 'cloud']

    def encode_and_decompose(self, z: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aligned = self.encoder(z)
        c, r = self.decomposition(aligned)
        y = self.recomposition(c, r)
        return aligned, c, r, y

    def forward(self, z_seg: Optional[torch.Tensor] = None,
                z_pose: Optional[torch.Tensor] = None,
                z_cloud: Optional[torch.Tensor] = None,
                visible_mask: Optional[torch.Tensor] = None) -> Dict:
        B = None
        device = None
        inputs = {'seg': z_seg, 'pose': z_pose, 'cloud': z_cloud}
        for z in inputs.values():
            if z is not None:
                B = z.shape[0]
                device = z.device
                break

        if B is None:
            raise ValueError("At least one modality must be provided")

        if z_cloud is not None:
            z_cloud = resample_cloud_tokens(z_cloud, self.num_parts)
            inputs['cloud'] = z_cloud

        if visible_mask is None:
            visible_mask = torch.zeros(B, 3, dtype=torch.bool, device=device)
            for i, modal in enumerate(self.modality_names):
                if inputs[modal] is not None:
                    visible_mask[:, i] = True
        else:
            visible_mask = visible_mask.to(device=device, dtype=torch.bool)

        for i, modal in enumerate(self.modality_names):
            if inputs[modal] is None:
                visible_mask[:, i] = False

        aligned = {}
        decomposed = {}
        y_dict = {}

        for i, modal in enumerate(self.modality_names):
            if inputs[modal] is not None:
                z_aligned, c, r, y = self.encode_and_decompose(inputs[modal])
                aligned[modal] = z_aligned
                decomposed[modal] = {'c': c, 'r': r, 'aligned': z_aligned}
                y_dict[modal] = y

        visible_modals = [m for m in self.modality_names if m in decomposed]
        if len(visible_modals) > 1:
            h_all = []
            for modal in self.modality_names:
                if modal in decomposed:
                    h_all.append(decomposed[modal]['c'].mean(dim=-1))
                else:
                    ref = next(iter(decomposed.values()))
                    B_ref, D_ref, _ = ref['c'].shape
                    h_all.append(torch.zeros(B_ref, D_ref, device=ref['c'].device, dtype=ref['c'].dtype))
            h_all = torch.stack(h_all, dim=1)

            ref_idx = visible_mask.int().argmax(dim=1)
            ref_onehot = F.one_hot(ref_idx, num_classes=len(self.modality_names)).to(h_all.dtype)
            h_ref = (h_all * ref_onehot.unsqueeze(-1)).sum(dim=1)

            for m_idx, modal in enumerate(self.modality_names):
                if modal not in decomposed:
                    continue
                h_modal = h_all[:, m_idx]
                dot = (h_ref * h_modal).sum(dim=-1, keepdim=True)
                sign = torch.where(dot >= 0,
                                   torch.ones_like(dot),
                                   -torch.ones_like(dot)).detach()
                sign = torch.where(visible_mask[:, m_idx].unsqueeze(-1), sign, torch.ones_like(sign))
                sign_3d = sign.unsqueeze(-1)
                decomposed[modal]['c'] = decomposed[modal]['c'] * sign_3d
                decomposed[modal]['r'] = decomposed[modal]['r'] * sign_3d
                decomposed[modal]['aligned'] = decomposed[modal]['aligned'] * sign_3d
                aligned[modal] = aligned[modal] * sign_3d
                y_dict[modal] = y_dict[modal] * sign_3d

        for m_idx, modal in enumerate(self.modality_names):
            if modal in decomposed:
                mask_3d = visible_mask[:, m_idx].float().view(-1, 1, 1)
                decomposed[modal]['c'] = decomposed[modal]['c'] * mask_3d
                decomposed[modal]['r'] = decomposed[modal]['r'] * mask_3d
                decomposed[modal]['aligned'] = decomposed[modal]['aligned'] * mask_3d
                aligned[modal] = aligned[modal] * mask_3d
                y_dict[modal] = y_dict[modal] * mask_3d

        c_list = []
        y_list = []
        for modal in self.modality_names:
            if modal in decomposed:
                c_list.append(decomposed[modal]['c'].permute(0, 2, 1))
                y_list.append(y_dict[modal].permute(0, 2, 1))
            else:
                ref = next(iter(decomposed.values()))
                B_ref, D_ref, P_ref = ref['c'].shape
                zeros = torch.zeros(B_ref, P_ref, D_ref, device=ref['c'].device, dtype=ref['c'].dtype)
                c_list.append(zeros)
                y_list.append(zeros)

        C_out = torch.stack(c_list, dim=1)
        Y_out = torch.stack(y_list, dim=1)

        if C_out is not None:
            mask_f = visible_mask.to(C_out.dtype)
            denom = mask_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1) * C_out.size(2)
            h = (C_out * mask_f[:, :, None, None]).sum(dim=[1, 2]) / denom
        else:
            h = torch.zeros(B, self.d_model, device=device)

        p_pred, w = self.reliability_head(h)

        visible_modalities_per_sample = []
        for b in range(B):
            mods = [m for i, m in enumerate(self.modality_names) if visible_mask[b, i]]
            visible_modalities_per_sample.append(mods)

        return {
            'aligned': aligned,
            'decomposed': decomposed,
            'y': y_dict,
            'C_out': C_out,
            'Y_out': Y_out,
            'h': h,
            'w': w,
            'p_pred': p_pred,
            'visible_mask': visible_mask,
            'visible_modalities_batch': visible_modals,
            'visible_modalities_per_sample': visible_modalities_per_sample,
        }

    def forward_with_imputation(self, z_seg: Optional[torch.Tensor] = None,
                                z_pose: Optional[torch.Tensor] = None,
                                z_cloud: Optional[torch.Tensor] = None,
                                missing_modalities: Optional[List[str]] = None
                                ) -> Dict:
        outputs = self.forward(z_seg, z_pose, z_cloud)

        if missing_modalities and len(outputs['visible_modalities_batch']) > 0:
            visible_tokens, key_padding_mask = self.imputer.prepare_visible_tokens(
                outputs['decomposed'],
                outputs['visible_mask'],
                self.modality_names
            )

            y_hat = self.imputer(visible_tokens, missing_modalities, key_padding_mask=key_padding_mask)
            outputs['y_hat'] = y_hat
        else:
            outputs['y_hat'] = {}

        return outputs

    def forward_teacher_student(self, z_seg: torch.Tensor,
                                z_pose: torch.Tensor,
                                z_cloud: Optional[torch.Tensor] = None,
                                drop_probs: Tuple[float, float, float] = (0.2, 0.7, 0.1),
                                two_modal_drop_probs: Tuple[float, float] = (0.3, 0.7)
                                ) -> Dict:
        available = []
        inputs = {'seg': z_seg, 'pose': z_pose, 'cloud': z_cloud}
        for modal in self.modality_names:
            if inputs[modal] is not None:
                available.append(modal)

        with torch.no_grad():
            teacher_out = self.forward(z_seg, z_pose, z_cloud)
            y_target = {m: y.detach() for m, y in teacher_out['y'].items()}

        B = z_seg.size(0)
        visible_mask = torch.zeros(B, 3, dtype=torch.bool, device=z_seg.device)
        for b in range(B):
            vis_set, _ = dropout_sampler(
                available, drop_probs, two_modal_drop_probs)
            for i, modal in enumerate(self.modality_names):
                if modal in vis_set:
                    visible_mask[b, i] = True

        student_out = self.forward(z_seg, z_pose, z_cloud, visible_mask=visible_mask)

        y_hat = {}
        missing_modalities_in_batch = [m for m in available if not visible_mask[:, self.modality_names.index(m)].all()]
        missing_mask = {m: ~visible_mask[:, i] for i, m in enumerate(self.modality_names) if m in available}

        if missing_modalities_in_batch:
            visible_tokens, key_padding_mask = self.imputer.prepare_visible_tokens(
                student_out['decomposed'],
                student_out['visible_mask'],
                self.modality_names
            )
            visible_tokens = visible_tokens.detach()
            y_hat = self.imputer(visible_tokens, missing_modalities_in_batch, key_padding_mask=key_padding_mask)

        return {
            'teacher': teacher_out,
            'student': student_out,
            'y_target': y_target,
            'y_hat': y_hat,
            'visible_set': student_out['visible_modalities_per_sample'],
            'missing_modalities_in_batch': missing_modalities_in_batch,
            'missing_set': missing_modalities_in_batch,
            'missing_mask': missing_mask,
            'visible_mask': visible_mask,
        }

    def compute_ortho_loss(self, decomposed: Dict,
                           visible_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        losses = []
        device = None

        for m_idx, modal in enumerate(self.modality_names):
            if modal in decomposed and decomposed[modal] is not None:
                c = decomposed[modal]['c']
                r = decomposed[modal]['r']
                device = c.device

                c_flat = c.reshape(c.size(0), -1)
                r_flat = r.reshape(r.size(0), -1)

                c_norm = F.normalize(c_flat, dim=-1)
                r_norm = F.normalize(r_flat, dim=-1)

                ortho_per_sample = (c_norm * r_norm).sum(dim=-1).pow(2)

                if visible_mask is not None:
                    mask = visible_mask[:, m_idx]
                    if mask.sum() > 0:
                        ortho = ortho_per_sample[mask].mean()
                    else:
                        continue
                else:
                    ortho = ortho_per_sample.mean()

                losses.append(ortho)

        if losses:
            return sum(losses) / len(losses)
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_final_representation(self, z_seg: Optional[torch.Tensor] = None,
                                 z_pose: Optional[torch.Tensor] = None,
                                 z_cloud: Optional[torch.Tensor] = None,
                                 impute_missing: bool = True
                                 ) -> Dict[str, torch.Tensor]:
        available = []
        missing = []
        inputs = {'seg': z_seg, 'pose': z_pose, 'cloud': z_cloud}

        for modal in self.modality_names:
            if inputs[modal] is not None:
                available.append(modal)
            else:
                missing.append(modal)

        outputs = self.forward(z_seg, z_pose, z_cloud)

        r_final = {m: outputs['decomposed'][m]['r']
                   for m in outputs['decomposed']
                   if isinstance(outputs['decomposed'][m], dict)}

        if impute_missing and missing and available:
            visible_tokens, key_padding_mask = self.imputer.prepare_visible_tokens(
                outputs['decomposed'],
                outputs['visible_mask'],
                self.modality_names
            )
            r_hat = self.imputer(visible_tokens, missing, key_padding_mask=key_padding_mask)
            r_final.update(r_hat)

        return r_final


AYCEModel = UnifiedEncoderV2
UnifiedEncoder = UnifiedEncoderV2
