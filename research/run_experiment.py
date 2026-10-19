"""Reproducible P0 baseline and P1 certificate feasibility pilot (not a paper result)."""
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
import numpy as np

from safecadence.data import synthetic_graph, load_tsv_dataset, audit_splits
from safecadence.transe import train_transe, TrainConfig
from safecadence.ranking import filtered_metrics
from safecadence.certificate import RadiusCertificate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=None,
                   help='Folder containing train.txt valid.txt test.txt. Omit for synthetic demo.')
    p.add_argument('--out', type=Path, default=Path('outputs/pilot.json'))
    p.add_argument('--steps', type=int, default=350)
    p.add_argument('--update-steps', type=int, default=45)
    p.add_argument('--queries', type=int, default=50, help='Number of old triples examined for protection')
    p.add_argument('--perturbations', type=int, default=15)
    p.add_argument('--max-eval', type=int, default=100)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()

    kg = load_tsv_dataset(args.data_dir) if args.data_dir else synthetic_graph(seed=args.seed)
    audit = audit_splits(kg)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(kg.train))
    split = max(1, int(len(perm)*0.75))
    old, incoming = kg.train[perm[:split]], kg.train[perm[split:]]
    # These are observed training facts only; never expose test labels to the optimizer/certificate.
    observed = np.concatenate([old, incoming])
    base_cfg = TrainConfig(steps=args.steps, seed=args.seed)
    start = perf_counter()
    E0, R0, meta = train_transe(old, kg.n_entities, kg.n_relations, base_cfg)
    old_train_time = perf_counter()-start
    initial = {
        'old_train_filtered': filtered_metrics(E0, R0, old, kg.all_true, args.max_eval),
        'new_train_filtered': filtered_metrics(E0, R0, incoming, kg.all_true, args.max_eval),
        'test_filtered': filtered_metrics(E0, R0, kg.test, kg.all_true, args.max_eval),
    }
    sample = old[rng.choice(len(old), min(len(old), args.queries), replace=False)]
    start = perf_counter()
    checker = RadiusCertificate(E0, R0, sample, observed, k=args.k)
    checker_build = perf_counter()-start

    # Candidate update deliberately trains on the new facts only. This is NOT the original Cadence.
    update_cfg = TrainConfig(dim=base_cfg.dim, steps=args.update_steps,
                             lr=0.007, seed=args.seed+1)
    start = perf_counter()
    E1, R1, update_meta = train_transe(incoming, kg.n_entities, kg.n_relations,
                                        update_cfg, init=(E0, R0))
    update_seconds = perf_counter()-start
    # 2x2 matched-step factorial comparison: replay (off/on) x anchor (off/on).
    # All four arms start at the same model and get the same batch size and steps.
    replay_idx = rng.choice(len(old), size=len(incoming), replace=True)
    replay_data = np.concatenate([incoming, old[replay_idx]])
    arms = {}
    arm_metrics = {}
    for label, use_replay, anchored in [
        ('finetune', False, False), ('anchor_only', False, True),
        ('replay_only', True, False), ('replay_and_anchor', True, True),
    ]:
        config = TrainConfig(dim=base_cfg.dim, steps=args.update_steps,
                             lr=0.007, seed=args.seed+1,
                             relation_anchor_strength=0.1 if anchored else 0.)
        st = perf_counter()
        if label == 'finetune':
            e_arm, r_arm = E1, R1
            secs = update_seconds
        else:
            e_arm, r_arm, _ = train_transe(replay_data if use_replay else incoming,
                                           kg.n_entities, kg.n_relations,
                                           config, init=(E0, R0))
            secs = perf_counter()-st
        arms[label] = (e_arm, r_arm)
        arm_metrics[label] = {
            'old': filtered_metrics(e_arm, r_arm, old, kg.all_true, args.max_eval),
            'new': filtered_metrics(e_arm, r_arm, incoming, kg.all_true, args.max_eval),
            'test': filtered_metrics(e_arm, r_arm, kg.test, kg.all_true, args.max_eval),
            'runtime_seconds': round(secs, 4),
            'same_update_steps': args.update_steps,
            'same_batch_size': config.batch_size,
            'same_negative_count': config.negatives,
        }
    dE, dR = E1-E0, R1-R0
    raw_cert = checker.verify(E1, R1, stop_early=False)
    raw_exact = checker.verify_exhaustive(E1, R1, stop_early=False)
    if raw_cert.safe != raw_exact.safe:
        raise AssertionError('Certification mismatch, unsafe to report results!')

    attempts = []
    chosen = None
    for alpha in [1., 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
        EA, RA = E0+alpha*dE, R0+alpha*dR
        info = checker.verify(EA, RA, stop_early=False)
        exact = checker.verify_exhaustive(EA, RA, stop_early=False)
        assert info.safe == exact.safe, 'Certificate inconsistent with exact verifier'
        attempts.append({'alpha': alpha, 'safe': info.safe,
                         'visited_fraction': round(info.inspected_candidates / max(info.full_candidates, 1), 5),
                         'certificate_seconds': round(info.checking_seconds, 6),
                         'exact_seconds': round(exact.checking_seconds, 6)})
        if info.safe:
            chosen = (alpha, EA, RA)
            break
    if chosen is None:
        chosen = (0., E0, R0)
    alpha, Esafe, Rsafe = chosen
    chosen_results = {
        'old_train_filtered': filtered_metrics(Esafe, Rsafe, old, kg.all_true, args.max_eval),
        'new_train_filtered': filtered_metrics(Esafe, Rsafe, incoming, kg.all_true, args.max_eval),
        'test_filtered': filtered_metrics(Esafe, Rsafe, kg.test, kg.all_true, args.max_eval),
    }
    unsafe_results = {
        'old_train_filtered': filtered_metrics(E1, R1, old, kg.all_true, args.max_eval),
        'new_train_filtered': filtered_metrics(E1, R1, incoming, kg.all_true, args.max_eval),
    }

    # Soundness and cost on random candidate parameter updates, not an entire CKGE stream.
    trials = []
    for i in range(args.perturbations):
        scale = 0.001 if i % 2 == 0 else 0.025
        # Sparse entity displacement, with a potentially shared relation displacement.
        delta_entities = np.zeros_like(E0)
        touched = rng.choice(len(E0), min(5, len(E0)), replace=False)
        delta_entities[touched] = rng.normal(0, scale, (len(touched), E0.shape[1]))
        delta_rel = rng.normal(0, scale, R0.shape)
        EA, RA = E0+delta_entities, R0+delta_rel
        fast = checker.verify(EA, RA, stop_early=False)
        exact = checker.verify_exhaustive(EA, RA, stop_early=False)
        assert fast.safe == exact.safe, 'Soundness check failed'
        trials.append({'safe': fast.safe,
                       'inspected_fraction': fast.inspected_candidates/max(fast.full_candidates, 1),
                       'fast_seconds': fast.checking_seconds,
                       'exhaustive_seconds': exact.checking_seconds})
    report = {
        'status': 'pilot_only_not_scientific_validation',
        'seed': args.seed,
        'data_audit': audit,
        'note': 'Original KG-Cadence source code/checkpoints not supplied; this is independent TransE + certifier.',
        'training': {**meta, 'old_training_seconds': round(old_train_time, 4),
                     'update_training_seconds': round(update_seconds, 4),
                     'update_loss': update_meta['loss_log']},
        'initial_metrics': initial,
        'matched_step_baselines': arm_metrics,
        'unsafe_update_metrics': unsafe_results,
        'chosen_safe_update_metrics': chosen_results,
        'certificate': {
            'k': args.k,
            'protected_queries': len(checker.protected),
            'already_unsatisfactory_queries': checker.not_initially_protected,
            'build_seconds': round(checker_build, 6),
            'kdtree_only_seconds': round(checker.build_seconds, 6),
            'raw_update_safe': raw_cert.safe,
            'raw_update_visited_fraction': round(raw_cert.inspected_candidates/max(raw_cert.full_candidates, 1), 5),
            'raw_update_fast_seconds': round(raw_cert.checking_seconds, 6),
            'raw_update_exhaustive_seconds': round(raw_exact.checking_seconds, 6),
            'chosen_step_alpha': alpha,
            'step_search': attempts,
            'random_trials': len(trials),
            'random_trial_safety_agreement': all(t['safe'] in (True, False) for t in trials),
            'trial_safe_rate': round(mean(t['safe'] for t in trials), 4) if trials else None,
            'trial_visited_fraction_mean': round(mean(t['inspected_fraction'] for t in trials), 5) if trials else None,
            'trial_fast_seconds_mean': round(mean(t['fast_seconds'] for t in trials), 6) if trials else None,
            'trial_exhaustive_seconds_mean': round(mean(t['exhaustive_seconds'] for t in trials), 6) if trials else None,
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(args.out.with_suffix('.npz'), old_entities=E0, old_relations=R0,
                        new_entities=E1, new_relations=R1, safe_entities=Esafe,
                        safe_relations=Rsafe)
    print(json.dumps({'report': str(args.out), 'audit': audit,
                      'metrics': initial,
                      'matched_step_baselines': {name: {'old_mrr': val['old']['mrr'], 'new_mrr': val['new']['mrr'], 'test_mrr': val['test']['mrr']} for name, val in arm_metrics.items()},
                      'protected_queries': len(checker.protected),
                      'raw_update_safe': raw_cert.safe,
                      'chosen_alpha': alpha,
                      'trial_visited_fraction': report['certificate']['trial_visited_fraction_mean'],
                      'trial_fast_seconds_mean': report['certificate']['trial_fast_seconds_mean'],
                      'trial_exhaustive_seconds_mean': report['certificate']['trial_exhaustive_seconds_mean']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
