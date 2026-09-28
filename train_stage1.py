import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import importlib.util


def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


embedding_dataset = import_module_from_path(
    'embedding_dataset',
    os.path.join(PROJECT_ROOT, 'opengait/data/embedding_dataset.py')
)
get_dataloader = embedding_dataset.get_dataloader
get_mixed_dataloader = embedding_dataset.get_mixed_dataloader

unified_encoder_module = import_module_from_path(
    'unified_encoder',
    os.path.join(PROJECT_ROOT, 'opengait/modeling/models/unified_encoder.py')
)
UnifiedEncoder = unified_encoder_module.UnifiedEncoder

geo_alignment = import_module_from_path(
    'geo_alignment',
    os.path.join(PROJECT_ROOT, 'opengait/modeling/losses/geo_alignment.py')
)
compute_total_loss = geo_alignment.compute_total_loss_v2


def get_config(dataset_name, base_path):
    if dataset_name == 'sustech1k':
        return {
            'name': 'SUSTech1K',
            'seg_root': f'{base_path}/SUSTech1K/Baseline/GaitBase_SUSTech1K_SilsAligned/embeddings',
            'pose_root': f'{base_path}/SUSTech1K/DeepGaitV2/DeepGaitV2/embeddings',
            'cloud_root': f'{base_path}/SUSTech1K/LidarGaitPlusPlus/lidargaitv2/embeddings',
            'num_modals': 3,
        }
    elif dataset_name == 'ccpg':
        return {
            'name': 'CCPG',
            'seg_root': f'{base_path}/CCPG/Baseline/GaitBase/embeddings',
            'pose_root': f'{base_path}/CCPG/DeepGaitV2/DeepGaitV2/embeddings',
            'cloud_root': None,
            'num_modals': 2,
        }
    elif dataset_name == 'mixed':
        return {
            'name': 'Mixed_CCPG_SUSTech1K',
            'datasets': [
                {
                    'name': 'CCPG',
                    'seg_root': f'{base_path}/CCPG/Baseline/GaitBase/embeddings',
                    'pose_root': f'{base_path}/CCPG/DeepGaitV2/DeepGaitV2/embeddings',
                    'cloud_root': None,
                },
                {
                    'name': 'SUSTech1K',
                    'seg_root': f'{base_path}/SUSTech1K/Baseline/GaitBase_SUSTech1K_SilsAligned/embeddings',
                    'pose_root': f'{base_path}/SUSTech1K/DeepGaitV2/DeepGaitV2/embeddings',
                    'cloud_root': f'{base_path}/SUSTech1K/LidarGaitPlusPlus/lidargaitv2/embeddings',
                },
            ],
            'num_modals': '2+3',
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def forward_batch(batch_data, model, cfg, use_amp=True):
    z_seg = batch_data['seg_emb'].cuda()
    z_pose = batch_data['pose_emb'].cuda()
    z_cloud = batch_data['cloud_emb']
    if z_cloud is not None:
        z_cloud = z_cloud.cuda()
    pid = batch_data['pid'].cuda()

    B = z_seg.shape[0]
    noise_scale = cfg['noise_scale']

    with autocast('cuda', enabled=use_amp):
        if noise_scale > 0:
            z_seg = z_seg + torch.randn_like(z_seg) * noise_scale
            z_pose = z_pose + torch.randn_like(z_pose) * noise_scale
            if z_cloud is not None:
                z_cloud = z_cloud + torch.randn_like(z_cloud) * noise_scale

        ts_out = model.forward_teacher_student(
            z_seg, z_pose, z_cloud,
            drop_probs=cfg['drop_probs'],
            two_modal_drop_probs=cfg['two_modal_drop_probs']
        )

        L_total, loss_dict = compute_total_loss(
            ts_out,
            person_ids=pid,
            lambda_geo=cfg['lambda_geo'],
            lambda_dir=cfg['lambda_dir'],
            lambda_ortho=cfg['lambda_ortho'],
            lambda_pred=cfg['lambda_pred'],
            lambda_de=cfg['lambda_de'],
            lambda_con=cfg['lambda_con'],
            lambda_private=cfg['lambda_private'],
        )

    return L_total, loss_dict, B


def train_step_single(batch, model, cfg, use_amp=True):
    return forward_batch(batch, model, cfg, use_amp)


def train_step_mixed(batch, model, cfg, use_amp=True):
    total_loss = 0
    total_samples = 0
    combined_losses = {
        'L_total': 0,
        'L_geo': 0,
        'L_dir': 0,
        'L_ortho': 0,
        'L_pred': 0,
        'L_de': 0,
        'L_con': 0,
        'L_private': 0,
    }

    if batch['batch_2m'] is not None:
        L_2m, losses_2m, B_2m = forward_batch(batch['batch_2m'], model, cfg, use_amp)
        total_loss = total_loss + L_2m * B_2m
        total_samples += B_2m
        for k in combined_losses:
            combined_losses[k] += losses_2m[k] * B_2m

    if batch['batch_3m'] is not None:
        L_3m, losses_3m, B_3m = forward_batch(batch['batch_3m'], model, cfg, use_amp)
        total_loss = total_loss + L_3m * B_3m
        total_samples += B_3m
        for k in combined_losses:
            combined_losses[k] += losses_3m[k] * B_3m

    if total_samples > 0:
        total_loss = total_loss / total_samples
        for k in combined_losses:
            combined_losses[k] /= total_samples

    return total_loss, combined_losses


def train_epoch(loader, model, optimizer, scaler, cfg, epoch, is_mixed=False, use_amp=True):
    model.train()

    epoch_losses = {
        'L_total': 0,
        'L_geo': 0,
        'L_dir': 0,
        'L_ortho': 0,
        'L_pred': 0,
        'L_de': 0,
        'L_con': 0,
        'L_private': 0,
    }
    n_batches = len(loader)

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        if is_mixed:
            L_total, loss_dict = train_step_mixed(batch, model, cfg, use_amp)
        else:
            L_total, loss_dict, _ = train_step_single(batch, model, cfg, use_amp)

        if use_amp:
            scaler.scale(L_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            L_total.backward()
            optimizer.step()

        for k, v in loss_dict.items():
            epoch_losses[k] += v

        if batch_idx % cfg['log_interval'] == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch+1}/{cfg['total_epochs']}] "
                  f"Batch [{batch_idx:4d}/{n_batches}] "
                  f"Loss: {loss_dict['L_total']:.4f} "
                  f"(geo: {loss_dict['L_geo']:.4f}, "
                  f"dir: {loss_dict['L_dir']:.4f}, "
                  f"ortho: {loss_dict['L_ortho']:.4f}, "
                  f"pred: {loss_dict['L_pred']:.4f}, "
                  f"con: {loss_dict['L_con']:.4f}, "
                  f"de: {loss_dict['L_de']:.4f}, "
                  f"private: {loss_dict['L_private']:.4f}) "
                  f"lr: {lr:.2e}")

    for k in epoch_losses:
        epoch_losses[k] /= n_batches

    elapsed = time.time() - start_time

    return epoch_losses, elapsed


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, cfg, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'cfg': cfg,
    }, save_path)

    print(f"Saved checkpoint to {save_path}")


def main(args):
    data_cfg = get_config(args.dataset, args.embedding_root)
    is_mixed = args.dataset == 'mixed'

    cfg = {
        'dataset': args.dataset,
        'is_mixed': is_mixed,

        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'num_parts': args.num_parts,
        'use_part_embed': args.use_part_embed,

        'lambda_geo': args.lambda_geo,
        'lambda_dir': args.lambda_dir,
        'lambda_ortho': args.lambda_ortho,
        'lambda_pred': args.lambda_pred,
        'lambda_de': args.lambda_de,
        'lambda_con': args.lambda_con,
        'lambda_private': args.lambda_private,

        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'total_epochs': args.total_epochs,
        'num_workers': args.num_workers,
        'use_amp': args.use_amp,
        'noise_scale': args.noise_scale,
        'drop_probs': tuple(args.drop_probs),
        'two_modal_drop_probs': tuple(args.two_modal_drop_probs),

        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'save_dir': args.save_dir or os.path.join('output', 'UnifiedEncoder', data_cfg['name']),
    }

    if args.dataset == 'ccpg':
        cfg['lambda_geo'] = 0.1
        cfg['lambda_dir'] = 0.1

    print("=" * 60)
    print("UnifiedEncoder Training")
    print("=" * 60)
    print(f"Dataset: {data_cfg['name']} ({data_cfg['num_modals']} modals)")
    print(f"Mixed training: {is_mixed}")
    print("=" * 60)

    print("\nLoading data...")
    if is_mixed:
        loader, dataset = get_mixed_dataloader(
            datasets_config=data_cfg['datasets'],
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers'],
        )
    else:
        loader, dataset = get_dataloader(
            seg_root=data_cfg['seg_root'],
            pose_root=data_cfg['pose_root'],
            cloud_root=data_cfg.get('cloud_root'),
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers'],
        )

    print(f"Total samples: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")

    print("\nBuilding model...")
    model = UnifiedEncoder(
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        num_parts=cfg['num_parts'],
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['total_epochs']
    )

    scaler = GradScaler('cuda') if cfg['use_amp'] else None

    start_epoch = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        scheduler.load_state_dict(resume_ckpt['scheduler'])
        if scaler is not None and resume_ckpt.get('scaler') is not None:
            scaler.load_state_dict(resume_ckpt['scaler'])
        start_epoch = int(resume_ckpt.get('epoch', 0))
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    os.makedirs(cfg['save_dir'], exist_ok=True)
    config_path = os.path.join(cfg['save_dir'], 'config.json')
    with open(config_path, 'w') as f:
        cfg_to_save = {k: v for k, v in cfg.items()}
        json.dump(cfg_to_save, f, indent=2)
    print(f"\nSaved config to {config_path}")

    print("\n" + "=" * 60)
    print("Start training...")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(start_epoch, cfg['total_epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['total_epochs']}")
        print("-" * 40)

        epoch_losses, elapsed = train_epoch(
            loader, model, optimizer, scaler, cfg, epoch,
            is_mixed=is_mixed, use_amp=cfg['use_amp']
        )

        scheduler.step()

        print(f"\n  Epoch {epoch + 1} Summary:")
        print(f"    L_total: {epoch_losses['L_total']:.4f}")
        print(f"    L_geo:   {epoch_losses['L_geo']:.4f}")
        print(f"    L_dir:   {epoch_losses['L_dir']:.4f}")
        print(f"    L_ortho: {epoch_losses['L_ortho']:.4f}")
        print(f"    L_pred:  {epoch_losses['L_pred']:.4f}")
        print(f"    L_con:   {epoch_losses['L_con']:.4f}")
        print(f"    L_de:    {epoch_losses['L_de']:.4f}")
        print(f"    L_priv:  {epoch_losses['L_private']:.4f}")
        print(f"    Time:    {elapsed:.1f}s")
        print(f"    LR:      {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % cfg['save_interval'] == 0:
            save_path = os.path.join(
                cfg['save_dir'],
                f'unified_encoder_epoch{epoch+1:03d}.pt'
            )
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, cfg, save_path)

        if epoch_losses['L_total'] < best_loss:
            best_loss = epoch_losses['L_total']
            save_path = os.path.join(cfg['save_dir'], 'unified_encoder_best.pt')
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, cfg, save_path)
            print(f"  New best loss: {best_loss:.4f}")

    save_path = os.path.join(cfg['save_dir'], 'unified_encoder_final.pt')
    save_checkpoint(model, optimizer, scheduler, scaler, cfg['total_epochs'], cfg, save_path)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {cfg['save_dir']}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mixed',
                        choices=['sustech1k', 'ccpg', 'mixed'])

    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_parts', type=int, default=16)
    parser.add_argument('--use_part_embed', action='store_true')

    parser.add_argument('--lambda_geo', type=float, default=1.0)
    parser.add_argument('--lambda_dir', type=float, default=1.0)
    parser.add_argument('--lambda_ortho', type=float, default=0.1)
    parser.add_argument('--lambda_pred', type=float, default=0.5)
    parser.add_argument('--lambda_de', type=float, default=0.0)
    parser.add_argument('--lambda_con', type=float, default=0.0)
    parser.add_argument('--lambda_private', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    parser.add_argument('--noise_scale', type=float, default=0.01)
    parser.add_argument('--drop_probs', type=float, nargs=3, default=[0.2, 0.7, 0.1])
    parser.add_argument('--two_modal_drop_probs', type=float, nargs=2, default=[0.3, 0.7])

    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--embedding_root', type=str, default='output')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    main(args)
