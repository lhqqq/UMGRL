import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def compute_L_geo_masked(C_out: torch.Tensor,
                         visible_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, M, P, D = C_out.shape
    device = C_out.device
    original_dtype = C_out.dtype

    if visible_mask is None:
        visible_mask = torch.ones(B, M, dtype=torch.bool, device=device)

    losses = []

    for b in range(B):
        vis_idx = visible_mask[b].nonzero(as_tuple=True)[0]
        n_visible = len(vis_idx)

        if n_visible < 2:
            continue

        C_vis = C_out[b, vis_idx]
        C_modal = C_vis.mean(dim=1)
        C_modal = F.normalize(C_modal, dim=-1)

        C_t = C_modal.T.float()
        _, S, _ = torch.linalg.svd(C_t, full_matrices=False)
        S = S.to(original_dtype)

        q = S[0] / (S.sum() + 1e-8)
        loss = -torch.log(q + 1e-8)
        losses.append(loss)

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=device, dtype=original_dtype)


def compute_L_geo(Z_out: torch.Tensor) -> torch.Tensor:
    B, M, P, C = Z_out.shape
    original_dtype = Z_out.dtype

    Z_modal = Z_out.mean(dim=2)
    Z_modal = F.normalize(Z_modal, dim=-1)

    Z_t = Z_modal.transpose(1, 2).float()
    _, S, _ = torch.linalg.svd(Z_t, full_matrices=False)
    S = S.to(original_dtype)

    q = S[:, 0] / (S.sum(dim=-1) + 1e-8)
    L_geo = (-torch.log(q + 1e-8)).mean()

    return L_geo


def compute_L_dir(Z_out: torch.Tensor,
                  visible_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, M, P, C = Z_out.shape
    device = Z_out.device

    if visible_mask is None:
        visible_mask = torch.ones(B, M, dtype=torch.bool, device=device)

    Z_modal = Z_out.mean(dim=2)
    Z_modal = F.normalize(Z_modal, dim=-1)

    total_sim = 0.0
    num_pairs = 0

    for b in range(B):
        vis_idx = visible_mask[b].nonzero(as_tuple=True)[0]
        n_visible = len(vis_idx)

        if n_visible < 2:
            continue

        for i in range(n_visible):
            for j in range(i + 1, n_visible):
                sim = (Z_modal[b, vis_idx[i]] * Z_modal[b, vis_idx[j]]).sum()
                total_sim = total_sim + sim
                num_pairs += 1

    if num_pairs > 0:
        avg_sim = total_sim / num_pairs
        return 1.0 - avg_sim

    return torch.tensor(0.0, device=device)


def compute_L_de(w_pred: torch.Tensor, Z_out: torch.Tensor) -> torch.Tensor:
    B, M, P, C = Z_out.shape
    original_dtype = Z_out.dtype

    Z_modal = Z_out.mean(dim=2)
    Z_modal = F.normalize(Z_modal, dim=-1)
    Z_t = Z_modal.transpose(1, 2).float()

    _, S, _ = torch.linalg.svd(Z_t, full_matrices=False)
    S = S.to(original_dtype)

    q_true = S[:, 0] / (S.sum(dim=-1) + 1e-8)

    L_de = F.mse_loss(w_pred, q_true.detach())

    return L_de


def compute_L_ortho(decomposed: Dict,
                    visible_mask: Optional[torch.Tensor] = None,
                    modality_names: List[str] = ['seg', 'pose', 'cloud']) -> torch.Tensor:
    ortho_losses = []
    device = None

    for m_idx, modal in enumerate(modality_names):
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
                    ortho_loss = ortho_per_sample[mask].mean()
                else:
                    continue
            else:
                ortho_loss = ortho_per_sample.mean()

            ortho_losses.append(ortho_loss)

    if ortho_losses:
        return sum(ortho_losses) / len(ortho_losses)
    return torch.tensor(0.0, device=device if device else 'cpu')


def compute_L_private(decomposed: Dict, mode: str = 'decorrelation') -> torch.Tensor:
    r_list = []
    for modal in ['seg', 'pose', 'cloud']:
        if modal in decomposed and decomposed[modal] is not None:
            r_list.append(decomposed[modal]['r'])

    if len(r_list) < 2:
        return torch.tensor(0.0, device=r_list[0].device if r_list else 'cpu')

    if mode == 'decorrelation':
        decorr_losses = []

        for i in range(len(r_list)):
            for j in range(i + 1, len(r_list)):
                r_i = r_list[i]
                r_j = r_list[j]

                B, D, P = r_i.shape

                r_i_flat = r_i.reshape(B, -1).T
                r_j_flat = r_j.reshape(B, -1).T

                r_i_centered = r_i_flat - r_i_flat.mean(dim=1, keepdim=True)
                r_j_centered = r_j_flat - r_j_flat.mean(dim=1, keepdim=True)

                cross_cov_diag = (r_i_centered * r_j_centered).mean(dim=1)

                std_i = r_i_centered.std(dim=1) + 1e-8
                std_j = r_j_centered.std(dim=1) + 1e-8
                cross_corr_diag = cross_cov_diag / (std_i * std_j)

                decorr_loss = cross_corr_diag.pow(2).mean()
                decorr_losses.append(decorr_loss)

        return sum(decorr_losses) / len(decorr_losses)

    elif mode == 'variance':
        var_losses = []

        for r in r_list:
            B, D, P = r.shape
            r_flat = r.reshape(B, -1)
            var = r_flat.var(dim=0)
            var_loss = F.relu(1.0 - var.sqrt()).mean()
            var_losses.append(var_loss)

        return sum(var_losses) / len(var_losses)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_L_pred(y_hat: Dict[str, torch.Tensor],
                   y_target: Dict[str, torch.Tensor],
                   loss_type: str = 'l2',
                   missing_mask: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    losses = []

    for modal in y_hat:
        if modal in y_target:
            pred = y_hat[modal]
            target = y_target[modal]

            if missing_mask is not None and modal in missing_mask:
                mask = missing_mask[modal].to(pred.device)
                if mask.sum() == 0:
                    continue
                pred = pred[mask]
                target = target[mask]

            if loss_type == 'l1':
                loss = F.l1_loss(pred, target)
            elif loss_type == 'smooth_l1':
                loss = F.smooth_l1_loss(pred, target)
            else:
                loss = F.mse_loss(pred, target)

            losses.append(loss)

    if losses:
        return sum(losses) / len(losses)
    else:
        device = next(iter(y_hat.values())).device if y_hat else 'cpu'
        return torch.tensor(0.0, device=device)


def compute_L_con(h1: torch.Tensor, h2: torch.Tensor,
                  person_ids: Optional[torch.Tensor] = None,
                  tau: float = 0.07) -> torch.Tensor:
    B = h1.shape[0]
    device = h1.device

    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)

    sim_matrix = torch.mm(h1, h2.T) / tau

    if person_ids is not None:
        id_matrix = person_ids.unsqueeze(0) == person_ids.unsqueeze(1)
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        false_neg_mask = id_matrix & ~diag_mask

        sim_matrix = sim_matrix.masked_fill(
            false_neg_mask,
            torch.finfo(sim_matrix.dtype).min
        )

    labels = torch.arange(B, device=device)
    loss_12 = F.cross_entropy(sim_matrix, labels)
    loss_21 = F.cross_entropy(sim_matrix.T, labels)

    return (loss_12 + loss_21) / 2


def compute_total_loss_v2(teacher_student_output: Dict,
                          person_ids: Optional[torch.Tensor] = None,
                          lambda_geo: float = 1.0,
                          lambda_dir: float = 1.0,
                          lambda_ortho: float = 0.1,
                          lambda_pred: float = 0.5,
                          lambda_de: float = 0.0,
                          lambda_con: float = 0.0,
                          lambda_private: float = 0.0,
                          ) -> Tuple[torch.Tensor, Dict]:
    student = teacher_student_output['student']
    y_hat = teacher_student_output['y_hat']
    y_target = teacher_student_output['y_target']
    visible_set = teacher_student_output['visible_set']
    missing_modalities_in_batch = teacher_student_output.get(
        'missing_modalities_in_batch',
        teacher_student_output.get('missing_set', [])
    )

    device = student['h'].device


    C_out = student['C_out']
    if C_out is not None and C_out.size(1) >= 2:
        visible_mask = student.get('visible_mask', None)
        if visible_mask is not None:
            L_geo = compute_L_geo_masked(C_out, visible_mask)
        else:
            L_geo = compute_L_geo(C_out)
    else:
        L_geo = torch.tensor(0.0, device=device)

    if C_out is not None and C_out.size(1) >= 2:
        visible_mask = student.get('visible_mask', None)
        L_dir = compute_L_dir(C_out, visible_mask)
    else:
        L_dir = torch.tensor(0.0, device=device)

    visible_mask = student.get('visible_mask', None)
    L_ortho = compute_L_ortho(student['decomposed'], visible_mask=visible_mask)

    missing_mask = teacher_student_output.get('missing_mask', None)
    if y_hat and missing_modalities_in_batch:
        y_target_missing = {m: y_target[m] for m in missing_modalities_in_batch if m in y_target}
        L_pred = compute_L_pred(y_hat, y_target_missing, loss_type='l2', missing_mask=missing_mask)
    else:
        L_pred = torch.tensor(0.0, device=device)


    if lambda_de > 0 and C_out is not None and C_out.size(1) >= 2:
        L_de = compute_L_de(student['w'], C_out)
    else:
        L_de = torch.tensor(0.0, device=device)

    L_con = torch.tensor(0.0, device=device)

    if lambda_private > 0:
        L_private = compute_L_private(student['decomposed'])
    else:
        L_private = torch.tensor(0.0, device=device)

    L_total = (lambda_geo * L_geo +
               lambda_dir * L_dir +
               lambda_ortho * L_ortho +
               lambda_pred * L_pred +
               lambda_de * L_de +
               lambda_con * L_con +
               lambda_private * L_private)

    loss_dict = {
        'L_total': L_total.item(),
        'L_geo': L_geo.item() if isinstance(L_geo, torch.Tensor) else L_geo,
        'L_dir': L_dir.item() if isinstance(L_dir, torch.Tensor) else L_dir,
        'L_ortho': L_ortho.item() if isinstance(L_ortho, torch.Tensor) else L_ortho,
        'L_pred': L_pred.item() if isinstance(L_pred, torch.Tensor) else L_pred,
        'L_de': L_de.item() if isinstance(L_de, torch.Tensor) else L_de,
        'L_con': L_con.item() if isinstance(L_con, torch.Tensor) else L_con,
        'L_private': L_private.item() if isinstance(L_private, torch.Tensor) else L_private,
        'visible_set': visible_set,
        'missing_modalities_in_batch': missing_modalities_in_batch,
        'missing_set': missing_modalities_in_batch,
    }

    return L_total, loss_dict


def compute_total_loss_decomposed(decomposed1, decomposed2, h1, h2, w1, w2,
                                  person_ids=None, **kwargs):
    C_out1 = decomposed1.get('C_out')
    C_out2 = decomposed2.get('C_out')

    if C_out1 is None or C_out2 is None:
        raise ValueError("C_out must be provided in decomposed dict")

    lambda_con = kwargs.get('lambda_con', 0.5)
    lambda_de = kwargs.get('lambda_de', 0.1)
    lambda_dir = kwargs.get('lambda_dir', 1.0)
    lambda_ortho = kwargs.get('lambda_ortho', 0.1)
    lambda_private = kwargs.get('lambda_private', 0.0)
    private_mode = kwargs.get('private_mode', 'decorrelation')

    L_geo = 0.5 * (compute_L_geo(C_out1) + compute_L_geo(C_out2))
    L_dir = 0.5 * (compute_L_dir(C_out1) + compute_L_dir(C_out2))
    L_con = compute_L_con(h1, h2, person_ids)
    L_de = 0.5 * (compute_L_de(w1, C_out1) + compute_L_de(w2, C_out2))
    L_ortho = 0.5 * (compute_L_ortho(decomposed1) + compute_L_ortho(decomposed2))

    if lambda_private > 0:
        L_private = 0.5 * (compute_L_private(decomposed1, mode=private_mode) +
                           compute_L_private(decomposed2, mode=private_mode))
    else:
        L_private = torch.tensor(0.0)

    L_total = (L_geo +
               lambda_dir * L_dir +
               lambda_con * L_con +
               lambda_de * L_de +
               lambda_ortho * L_ortho +
               lambda_private * L_private)

    loss_dict = {
        'L_total': L_total.item(),
        'L_geo': L_geo.item(),
        'L_dir': L_dir.item(),
        'L_con': L_con.item(),
        'L_de': L_de.item(),
        'L_ortho': L_ortho.item() if isinstance(L_ortho, torch.Tensor) else L_ortho,
        'L_private': L_private.item() if isinstance(L_private, torch.Tensor) else L_private,
    }

    return L_total, loss_dict
