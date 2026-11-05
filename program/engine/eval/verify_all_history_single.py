"""Independent dataset, fresh initialization, and isolated runtime audit."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest import mock

import cv2
import numpy as np
import torch

import detect_frame
from screen_corner_single import SingleModelPredictor, patch_boxes, sample_corner_patches
from training.screen_corner_ablation import read_rows, write_json, fingerprint


def verify_data(root):
    spec = json.loads((root / 'data/manifest.json').read_text())
    parts = {k: read_rows(root / f'data/{k}.jsonl') for k in ['train', 'validation', 'holdout']}
    paths = {k: {r['image_path'] for r in v} for k, v in parts.items()}
    hashes = {k: {r['image_sha256'] for r in v} for k, v in parts.items()}
    groups = {k: {r['content_group'] for r in v} for k, v in parts.items()}
    for key in ['validation', 'holdout']:
        assert not paths['train'] & paths[key]
        assert not hashes['train'] & hashes[key]
        assert not groups['train'] & groups[key]
    for p, digest in spec['inputs'].items():
        assert fingerprint(p) == digest, p
    allrows = read_rows(root / 'data/all_manual.jsonl')
    for i, row in enumerate(allrows):
        assert fingerprint(row['image_path']) == row['image_sha256'], row['image_path']
        if i % 500 == 0:
            print(f'source digest verification {i+1}/{len(allrows)}', flush=True)
    train_samples = read_rows(root / 'data/train_sampling.jsonl')
    used = Counter(r['image_sha256'] for r in train_samples)
    assert set(used) == hashes['train']
    assert all(1 <= n <= 4 for n in used.values())
    return {'manual_pages': len(allrows), 'source_images_unchanged': True, 'historical_input_files_unchanged': True,
            'train_evaluation_path_overlap': 0, 'train_evaluation_exact_photo_overlap': 0,
            'train_evaluation_content_group_overlap': 0, 'train_sampling_covers_all_unique_pages_each_epoch': True,
            'sampling_repeats': dict(Counter(used.values())),
            'validation_holdout_shared_content_groups': len(groups['validation'] & groups['holdout']),
            'regression_interpretation': 'historical observed regression; not independent blind test'}


def verify_model(root):
    cfg = json.loads((root / 'manifest.json').read_text())
    frozen = json.loads((root / 'reports/frozen_selection.json').read_text())
    initial = torch.load(root / 'artifacts/fresh_initialization.pt', map_location='cpu')
    artifact = Path(frozen['artifact']); assert fingerprint(artifact) == frozen['sha256']
    rows = read_rows(root / 'data/validation.jsonl')
    # Deterministic coverage: one page per source, first three source projects.
    picked = []; projects = set()
    for row in rows:
        if row['project_key'] not in projects:
            picked.append(row); projects.add(row['project_key'])
        if len(picked) == 3:
            break
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'screen_corner_single.pt'; shutil.copyfile(artifact, path)
        with (mock.patch.dict(os.environ, {'SCREEN_PDF_SINGLE_MODEL': str(path), 'SCREEN_PDF_MODEL_DIR': directory}),
              mock.patch('torch.load', side_effect=AssertionError('External weights read')),
              mock.patch('detect_frame.run_teacher_detection', side_effect=AssertionError('Old runtime called')),
              mock.patch('detect_frame.detect_best_candidate', side_effect=AssertionError('External candidate called'))):
            predictor = SingleModelPredictor(path)
            encoder_parameters = dict(predictor.model.encoder.named_parameters())
            changed_encoder = [k for k, v in encoder_parameters.items() if not torch.equal(v, initial['encoder'][k])]
            changed_global = [k for k, v in predictor.model.global_head.named_parameters() if not torch.equal(v, initial['global_head'][k])]
            changed_local = [k for k, v in predictor.model.local_head.named_parameters() if not torch.equal(v, initial['local_head'][k])]
            assert changed_encoder and changed_global and changed_local
            checks = []
            for row in picked:
                image = cv2.imread(row['image_path']); prediction = predictor.predict(image)
                payload = detect_frame.build_detect_payload(row['image_path'], image)
                assert payload['best']['source'] == 'runtime_single'
                assert payload['best']['method'] == 'screen_corner_single'
                assert len(payload['candidates']) == 1
                np.testing.assert_allclose(prediction['final_quad'], payload['best']['quad'], atol=1e-3)
                rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())[None].float() / 255.
                with torch.inference_mode():
                    coarse, _, _, feature = predictor.model.global_prediction(rgb)
                    boxes = patch_boxes(coarse)
                    relative, _ = predictor.model.local_head(sample_corner_patches(rgb, boxes, 64), sample_corner_patches(feature, boxes, 16))
                    final = (boxes[..., :2] + relative.view(-1, 4, 2) * (boxes[..., 2:] - boxes[..., :2])).clamp(0., 1.)
                    torch.testing.assert_close(predictor.model(rgb)['final_quad'], final)
                checks.append({'image_path': row['image_path'], 'source': payload['best']['source'], 'finite': bool(np.isfinite(prediction['final_quad']).all())})
            assert sorted(p.name for p in Path(directory).iterdir()) == ['screen_corner_single.pt']
        detect_frame._SINGLE_MODEL_RUNTIME = None
    return {'single_file_only': True, 'model_id': frozen['model_id'], 'artifact_sha256': frozen['sha256'],
            'encoder_parameter_tensors_changed': len(changed_encoder), 'global_parameter_tensors_changed': len(changed_global),
            'local_parameter_tensors_changed': len(changed_local), 'all_three_components_retrained': True,
            'isolated_detection': checks, 'cached_crop_operations_match_full_graph': True,
            'task_checkpoint_initialization': cfg['task_checkpoint_initialization'], 'teacher_models': cfg['teacher_models']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-only', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'dataset': verify_data(args.root)}
    if not args.data_only:
        report['artifact'] = verify_model(args.root)
    write_json(args.root / 'reports' / ('data_verification.json' if args.data_only else 'final_verification.json'), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
