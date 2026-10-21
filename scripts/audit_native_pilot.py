"""Independently audit completed native pilot artifacts without games or training.

Run in the pinned private Python environment. Outputs contain private provenance
and must remain private. Stored targets are not recomputed from the teacher.
Inputs must be trusted local artifacts. Owned frame and optimizer files use
torch.load with weights_only=False.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

VERSION = "86-tcg-agent/native-pilot-auditor/1"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


sha = digest = file_sha


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def tool_identity():
    return {"audit_tool_version": VERSION, "audit_tool_sha256": file_sha(__file__),
            "audit_runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                              "cpu_threads": torch.get_num_threads(), "linux_nice": os.nice(0)},
            "no_teacher_loaded_by_auditor": True, "no_games_started_by_auditor": True,
            "no_optimizer_updates_by_auditor": True}

def audit_stage1(args):
    BASE = args.stage1_run
    PRIVATE = args.output
    started = time.monotonic()
    def read(name):
        return json.loads((BASE / name).read_bytes())
    bindings_path = args.stage1_bindings
    bindings = json.loads(bindings_path.read_bytes())
    verified_sources = 0
    for path, expected in bindings['artifacts'].items():
        assert digest(path) == expected, ('source changed', path)
        verified_sources += 1
    os.environ['FASTENC_LIB'] = bindings['engine_library']
    native.configure_mew(bindings['mew_root'])
    result = read('training-result.json')
    update_count = result['updates']
    manifest = read('manifest.json')
    assert hashlib.sha256(canonical(bindings['artifacts'])).hexdigest() == bindings['artifact_manifest_sha256'] == manifest['artifact_manifest_sha256']
    protocol = manifest['protocol']
    initial_manifest, final_manifest = read('initial.json'), read('student.json')
    initial = native.load_student(BASE / 'initial.pt', expected_sha256=initial_manifest['weights_sha256'])
    final = native.load_student(BASE / 'student.pt', expected_sha256=final_manifest['weights_sha256'])
    assert final_manifest['weights_sha256'] == result['student_weights_sha256']
    assert initial_manifest['weights_sha256'] == result['initial_weights_sha256']
    assert dataclasses.asdict(initial.config) == dataclasses.asdict(final.config)
    fresh = native.build_student(native.SmallNativeConfig(**protocol['student']), seed=protocol['seeds']['student_init'])
    assert native.state_digest(fresh) == native.state_digest(initial)
    initial_params, final_params = dict(initial.named_parameters()), dict(final.named_parameters())
    critic_names = [name for name in initial_params if name.startswith('value_head.')]
    actor_changes = [name for name in initial_params if not name.startswith('value_head.') and not torch.equal(initial_params[name], final_params[name])]
    assert critic_names and actor_changes
    assert all(torch.equal(initial_params[name], final_params[name]) for name in critic_names)
    assert all(not final_params[name].requires_grad for name in critic_names)
    initial_buffers, final_buffers = dict(initial.named_buffers()), dict(final.named_buffers())
    assert initial_buffers.keys() == final_buffers.keys()
    assert all(torch.equal(value, final_buffers[name]) for name, value in initial_buffers.items())
    assert native.state_digest(initial) == initial_manifest['model_identity']['state_digest']
    assert native.state_digest(final) == final_manifest['model_identity']['state_digest']

    all_labels = []
    label_hashes = {}
    frame_ids = set()
    for index in range(1, update_count + 1):
        name = f'labels-{index:04d}.pt'
        labels = torch.load(BASE / name, map_location='cpu', weights_only=False)
        assert len(labels) == 32
        label_hashes[name] = digest(BASE / name)
        all_labels.extend(labels)
    probes = torch.load(BASE / 'heldout-probe-frames.pt', map_location='cpu', weights_only=False)
    assert len(probes) == 32
    target_entropy = []
    for frame, target in all_labels + probes:
        assert isinstance(frame, native.ActorFrame) and frame.real_rows == 1
        assert len(frame.records[0].eligible_columns) > 1
        assert frame.schema_digest == native._schema_digest()
        assert target.ndim == 1 and not target.requires_grad and target.grad_fn is None
        legal = torch.zeros_like(target, dtype=torch.bool)
        legal[list(frame.records[0].eligible_columns)] = True
        assert torch.isfinite(target).all() and (target >= 0).all()
        assert torch.all(target[~legal] == 0)
        assert abs(float(target.sum()) - 1) < 1e-5
        assert int(torch.count_nonzero(target)) == len(frame.records[0].eligible_columns)
        assert all(type(item.data) is bytes for item in frame._arrays)
        h = hashlib.sha256()
        h.update(json.dumps([frame.schema_digest, frame._seq_len, [dataclasses.asdict(r) for r in frame.records]], sort_keys=True).encode())
        for item in frame._arrays:
            h.update(json.dumps([item.name, item.dtype, item.shape]).encode())
            h.update(item.data)
        assert h.hexdigest() == frame.digest
        frame_ids.add(frame.digest)
        q = target[legal]
        target_entropy.append(float(-(q * q.log()).sum()))
    training_frame_ids = {frame.digest for frame, _ in all_labels}
    probe_overlap = [frame.digest for frame, _ in probes if frame.digest in training_frame_ids]

    recomputed = {}
    for name, model, receipt_name in [('initial', initial, 'probe-before.json'), ('final', final, 'probe-after.json')]:
        receipt = read(receipt_name)
        assert [row['frame_digest'] for row in receipt['rows']] == [frame.digest for frame, _ in probes]
        cross_entropies, agreements, kls = [], [], []
        max_error = 0.0
        for (frame, target), recorded in zip(probes, receipt['rows']):
            scores = native.score_policy(model, frame)
            legal = scores.legal_mask[0]
            q = target[legal]
            ce = float(-(q * scores.logits[0][legal].log_softmax(-1)).sum())
            entropy = float(-(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum())
            agreement = int(scores.logits[0].argmax() == target.argmax())
            max_error = max(max_error, abs(ce - recorded['cross_entropy']), abs(ce - entropy - recorded['forward_kl']))
            assert agreement == recorded['top1_agreement']
            cross_entropies.append(ce)
            agreements.append(agreement)
            kls.append(ce - entropy)
        assert max_error < 1e-6
        recomputed[name] = {'frames': len(probes), 'cross_entropy': sum(cross_entropies)/32,
                            'forward_kl': sum(kls)/32, 'top1_agreement': sum(agreements)/32,
                            'max_recorded_metric_absolute_error': max_error}

    episodes = [json.loads(line) for line in (BASE / 'training-episodes.jsonl').read_bytes().splitlines()]
    probe_episodes = [json.loads(line) for line in (BASE / 'probe-episodes.jsonl').read_bytes().splitlines()]
    updates = [json.loads(line) for line in (BASE / 'updates.jsonl').read_bytes().splitlines()]
    assert len(episodes) == result['episodes'] and len(updates) == update_count and update_count > 0
    assert sum(row['labels'] for row in episodes) == len(all_labels)
    assert sum(row['labels'] for row in probe_episodes) == 32
    assert sum(sum(choice['learner'] for choice in row['choices']) for row in episodes + probe_episodes) == result['budget']['admitted_learner_microsteps']
    assert result['budget']['teacher_queries'] == len(all_labels) + len(probes)
    for index, row in enumerate(updates, 1):
        assert row['labels_sha256'] == label_hashes[f'labels-{index:04d}.pt']
        assert row['examples'] == 32
    for index in range(len(episodes)):
        rng = torch.load(BASE / f'episode-{index:06d}-rng.pt', map_location='cpu', weights_only=True)
        assert set(rng) == {'learner', 'opponent'}
        assert all(value.dtype == torch.uint8 and value.ndim == 1 for value in rng.values())
    checkpoint_hashes = {}
    for name in ['initial'] + [f'update-{index:04d}' for index in range(1, update_count + 1)] + ['student']:
        receipt = read(name + '.json')
        assert digest(BASE / (name + '.pt')) == receipt['weights_sha256']
        assert receipt['protocol_sha256'] == manifest['protocol_sha256']
        state = torch.load(BASE / (name + '.state.pt'), map_location='cpu', weights_only=False)
        assert state['protocol_sha256'] == manifest['protocol_sha256']
        expected_step = 0 if name == 'initial' else (update_count if name == 'student' else int(name.removeprefix('update-')))
        optimizer_steps = [float(item['step']) for item in state['optimizer']['state'].values()]
        assert all(step == expected_step for step in optimizer_steps)
        assert bool(optimizer_steps) == bool(expected_step)
        checkpoint_hashes[name] = receipt['weights_sha256']
    native_counters = Counter()
    for row in episodes + probe_episodes:
        native_counters.update(row['native_stats'])
    receipt = {'schema':'86-tcg-agent/stage1-independent-audit/1', 'status':'passed',
               'source_files_verified':verified_sources, 'bindings_sha256':digest(bindings_path),
               'training_result_sha256':digest(BASE/'training-result.json'), 'manifest_sha256':digest(BASE/'manifest.json'),
               'initial_checkpoint_sha256':checkpoint_hashes['initial'], 'final_checkpoint_sha256':checkpoint_hashes['student'],
               'checkpoint_hashes':checkpoint_hashes, 'checkpoint_and_rng_states_loaded':update_count + 2, 'episode_rng_states_loaded':len(episodes),
               'fresh_initial_state_exact':True, 'architecture_equal':True, 'changed_actor_parameter_tensors':len(actor_changes),
               'critic_parameter_tensors_unchanged':len(critic_names), 'static_buffer_tensors_unchanged':len(initial_buffers),
               'labels':{'training':len(all_labels),'heldout':len(probes),'all_genuine':True,'full_positive_legal_support':True,
                         'unit_mass_tolerance':1e-5,'immutable_frame_bytes_and_digests_verified':True,
                         'probe_frames_also_present_in_optimizer_labels':len(probe_overlap),
                         'distinct_overlapping_frame_digests':len(set(probe_overlap))},
               'probe_recomputed':recomputed, 'training_episode_statuses':dict(Counter(row['status'] for row in episodes)),
               'probe_episode_statuses':dict(Counter(row['status'] for row in probe_episodes)),
               'native_counters_training_and_probe':dict(native_counters), 'budget':result['budget'],
               'audit_wall_seconds':time.monotonic()-started, 'no_teacher_loaded':True,'no_games_started':True,'no_optimizer_updates':True,
               'limitations':['Independent probe seed domains do not guarantee distinct observable frames. Exact stored frame overlap with optimizer labels is reported explicitly.',
                              'Stored teacher targets were checked for legality and probability validity, not recomputed from the teacher.',
                              'Teacher runtime immutability is a trainer attestation. This post-run audit does not load or observe the teacher model.',
                              'Frame digests establish immutable stored bytes, not independently reconstructed native actor visibility.',
                              'Checkpoint RNG states were deserialized and identified, not used for optimizer or active-game resume.',
                              'Probe imitation improvement does not establish playing strength or current Standard readiness.']}

    receipt.update(tool_identity())
    return receipt


def audit_probe_novelty(args):
    root = args.stage1_run
    private = args.output
    training_ids = set()
    update_count = json.loads((root/'training-result.json').read_bytes())['updates']
    for index in range(1, update_count + 1):
        training_ids.update(frame.digest for frame,target in torch.load(root/f'labels-{index:04d}.pt', weights_only=False))
    probes = torch.load(root/'heldout-probe-frames.pt', weights_only=False)
    overlap = [frame.digest for frame,target in probes if frame.digest in training_ids]
    novel = [(frame,target) for frame,target in probes if frame.digest not in training_ids]
    metrics={}
    for name in ('initial','student'):
        manifest=json.loads((root/(name+'.json')).read_bytes())
        model=native.load_student(root/(name+'.pt'), expected_sha256=manifest['weights_sha256'])
        rows=[]
        for frame,target in novel:
            scores=native.score_policy(model,frame)
            mask=scores.legal_mask[0]
            q=target[mask]
            ce=float(-(q*scores.logits[0][mask].log_softmax(-1)).sum())
            ent=float(-(q*q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum())
            rows.append({'frame_digest':frame.digest,'cross_entropy':ce,'forward_kl':ce-ent,
                         'top1_agreement':int(scores.logits[0].argmax()==target.argmax())})
        metrics[name]={'checkpoint_sha256':sha(root/(name+'.pt')),'count':len(rows),
                       **{key:sum(row[key] for row in rows)/len(rows) if rows else None for key in ('cross_entropy','forward_kl','top1_agreement')},'rows':rows}
    receipt={'schema':'86-tcg-agent/stage1-probe-novelty/1','status':'passed','original_probe_frames':len(probes),
             'exact_digest_disjoint_frames':len(novel),'overlap_frames':len(overlap),'overlap_digests':overlap,'metrics':metrics,
             'split':'Post hoc partition by exact immutable frame digest identity only. No outcome, loss or agreement used to select frames. Original checkpoints and original 32-frame metrics unchanged.',
             'source_audit_sha256':sha(private/'stage1-audit.json'),
             'limitations':['Exact digest novelty does not imply strategically different states or out-of-distribution generalization.','Imitation diagnostics do not establish playing strength.'],
             'no_games_started':True,'no_optimizer_updates':True,'no_teacher_loaded':True}

    receipt.update(tool_identity())
    return receipt


def audit_actorcritic(args):
    ROOT = args.actorcritic_run
    PRIVATE = args.output
    from tcg import native_actorcritic as implementation
    def read(name):
        return json.loads((ROOT / name).read_bytes())
    def rows(name):
        return [json.loads(line) for line in (ROOT / name).read_bytes().splitlines()]
    assert (ROOT/'actorcritic-result.json').is_file() and not (ROOT/'failure.json').exists()
    final_receipt=read('actorcritic-result.json')
    assert final_receipt['status']=='completed_bounded_pipeline'
    bindings_path=args.continuation_bindings
    bindings=json.loads(bindings_path.read_bytes())
    for path,expected in bindings['artifacts'].items():assert sha(path)==expected,('changed_source',path)
    assert sha(bindings['stage1_bindings'])==bindings['stage1_bindings_sha256']
    prior_bindings=json.loads(Path(bindings['stage1_bindings']).read_bytes())
    for key in ('engine_sha256','deck_manifest_sha256'):assert prior_bindings[key]==bindings[key]
    native.configure_mew(bindings['mew_root'])
    inputs=read('inputs.json')
    stage3=read('stage3-result.json')
    stage4=read('stage4-result.json')
    assert stage4['updates']==2
    models={}
    checkpoint_identities={}
    epoch_counts = [len(report['epochs']) for report in rows('stage4-updates.jsonl')]
    expected_steps = {'stage1-actor': 0, 'stage3': 1, 'stage4-update-0001': epoch_counts[0],
                      'stage4-update-0002': sum(epoch_counts), 'stage4': sum(epoch_counts)}
    for name in ('stage1-actor','stage3','stage4-update-0001','stage4-update-0002','stage4'):
        receipt=read(name+'.json')
        assert sha(ROOT/(name+'.pt'))==receipt['weights_sha256']
        assert sha(ROOT/(name+'.state.pt'))==receipt['state_sha256']
        state=torch.load(ROOT/(name+'.state.pt'),weights_only=False,map_location='cpu')
        assert state['lineage']['bindings_sha256']==sha(bindings_path)
        model=native.load_student(ROOT/(name+'.pt'),expected_sha256=receipt['weights_sha256'])
        assert native.state_digest(model)==receipt['identity']['state_digest']
        models[name]=model
        optimizer_steps=[] if state['optimizer'] is None else [float(item['step']) for item in state['optimizer']['state'].values()]
        assert all(step == expected_steps[name] for step in optimizer_steps)
        assert bool(optimizer_steps) == bool(expected_steps[name])
        checkpoint_identities[name]={'weights_sha256':receipt['weights_sha256'],'state_sha256':receipt['state_sha256'],
                                      'optimizer_state_entries':len(optimizer_steps),'optimizer_step_min':min(optimizer_steps) if optimizer_steps else None,
                                      'optimizer_step_max':max(optimizer_steps) if optimizer_steps else None}
    initial,calibrated,final=models['stage1-actor'],models['stage3'],models['stage4']
    assert native.state_digest(initial)==json.loads((args.stage1_run/'student.json').read_bytes())['model_identity']['state_digest']
    assert native.state_digest(models['stage4-update-0002'])==native.state_digest(final)
    initial_params=dict(initial.named_parameters())
    calibrated_params=dict(calibrated.named_parameters())
    final_params=dict(final.named_parameters())
    actor_names=[name for name in initial_params if not name.startswith('value_head.')]
    critic_names=[name for name in initial_params if name.startswith('value_head.')]
    assert all(torch.equal(initial_params[name],calibrated_params[name]) for name in actor_names)
    critic_changed=[name for name in critic_names if not torch.equal(initial_params[name],calibrated_params[name])]
    assert critic_changed
    stage4_actor_changed=[name for name in actor_names if not torch.equal(calibrated_params[name],final_params[name])]
    assert stage4_actor_changed
    buffer_count=0
    for name,value in initial.named_buffers():
        assert torch.equal(value,dict(calibrated.named_buffers())[name])
        assert torch.equal(value,dict(final.named_buffers())[name])
        buffer_count+=1

    data=torch.load(ROOT/'stage3-data.pt',weights_only=False,map_location='cpu')
    assert sha(ROOT/'stage3-data.pt')==stage3['data_sha256']
    train=data['training_examples'];probes=data['heldout_examples']
    assert [target for frame,target in train]==[row['outcome'] for row in rows('calibration-episodes.jsonl') for _ in range(row['retained_learner_steps'])]
    assert [target for frame,target in probes]==[row['outcome'] for row in rows('calibration_heldout-episodes.jsonl') for _ in range(row['retained_learner_steps'])]
    constant=math.fsum(target for frame,target in train)/len(train)
    calibration={}
    def predict(model,frame):
        with torch.no_grad():logits,value=model(frame._payload(),validate=True)
        return logits,float(value.tanh()[0])
    for key,model in [('heldout_before',initial),('heldout_after',calibrated)]:
        expected=stage3[key]
        assert len(probes)==expected['count']==128
        predictions=[];maximum=0.0
        for (frame,target),recorded in zip(probes,expected['rows']):
            assert frame.digest==recorded['frame_digest'] and target==recorded['target']
            logits,value=predict(model,frame)
            initial_logits,_=predict(initial,frame)
            assert torch.equal(logits,initial_logits)
            predictions.append((value,target))
            maximum=max(maximum,abs(value-recorded['prediction']))
        mse=math.fsum((value-target)**2 for value,target in predictions)/len(predictions)
        constant_mse=math.fsum((constant-target)**2 for _,target in predictions)/len(predictions)
        assert abs(mse-expected['mse'])<1e-6 and abs(constant_mse-expected['constant_mse'])<1e-6
        assert maximum<1e-6
        calibration[key]={'mse':mse,'constant_mse':constant_mse,'constant_training_mean':constant,'examples':len(predictions),'maximum_recorded_prediction_error':maximum}

    reports=rows('stage4-updates.jsonl')
    episode_rows=rows('stage4-episodes.jsonl')
    assert len(reports)==2 and len(episode_rows)==4
    ppo=[]
    for index,report in enumerate(reports,1):
        model=models['stage3' if index==1 else 'stage4-update-0001']
        path=ROOT/f'stage4-update-{index:04d}-episodes.pt'
        episodes=torch.load(path,weights_only=False,map_location='cpu')
        assert sha(path)==report['checkpoint']['lineage']['data_sha256']
        assert len(episodes)==2
        independent_adv=[];independent_returns=[];steps=[];max_logp=0.0;max_value=0.0
        for episode in episodes:
            assert episode.policy_digest==native.state_digest(model)
            matching=[row for row in episode_rows if str(row['engine_seed'])==episode.episode_id]
            assert len(matching)==1 and matching[0]['status']=='terminal'
            observed=matching[0]
            outcome=0.0 if observed['winner']==2 else (1.0 if observed['winner']==episode.learner_seat else -1.0)
            assert outcome==episode.outcome and observed['outcome']==outcome
            td=[]
            for t,step in enumerate(episode.steps):
                assert step.frame.records[0].select_player==episode.learner_seat
                assert step.action in step.frame.records[0].eligible_columns
                assert step.genuine==(len(step.frame.records[0].eligible_columns)>1)
                logits,value=predict(model,step.frame)
                logp=float(logits[0].log_softmax(-1)[step.action])
                max_logp=max(max_logp,abs(logp-step.old_logp));max_value=max(max_value,abs(value-step.old_value))
                next_value=episode.steps[t+1].old_value if t+1<len(episode.steps) else 0.0
                td.append((outcome if t+1==len(episode.steps) else 0.0)+next_value-step.old_value)
            for t in range(len(td)):
                independent_adv.append(math.fsum((0.95**(k-t))*td[k] for k in range(t,len(td))))
                independent_returns.append(outcome)
            steps.extend(episode.steps)
        genuine=[adv for step,adv in zip(steps,independent_adv) if step.genuine]
        mean=math.fsum(genuine)/len(genuine)
        sd=math.sqrt(math.fsum((adv-mean)**2 for adv in genuine)/len(genuine))
        normalized=[(adv-mean)/(sd+1e-8) if step.genuine else 0.0 for step,adv in zip(steps,independent_adv)]
        actual_steps,actual_norm,actual_returns=implementation.terminal_targets(episodes)
        max_norm=max(abs(a-b) for a,b in zip(normalized,actual_norm))
        assert max_norm<1e-10 and tuple(independent_returns)==actual_returns
        assert max_logp<1e-6 and max_value<1e-6
        ppo.append({'update':index,'episodes':2,'learner_steps':len(steps),'genuine_steps':len(genuine),
                    'max_old_logp_reload_error':max_logp,'max_old_value_reload_error':max_value,
                    'max_normalized_gae_error':max_norm,'MC_returns_exact':True,'epochs':len(report['epochs']),
                    'data_sha256':sha(path)})
    statuses={}
    rng_count=0
    native_counters=Counter()
    for domain in ('calibration','calibration_heldout','stage4'):
        observed=rows(domain+'-episodes.jsonl')
        statuses[domain]=dict(Counter(row['status'] for row in observed))
        assert all(row['status']=='terminal' for row in observed)
        for row in observed:
            native_counters.update(row['native_stats'])
            rng=torch.load(ROOT/f'{domain}-{row["index"]:04d}.rng.pt',weights_only=True,map_location='cpu')
            assert set(rng)=={'learner','opponent'}
            assert all(value.dtype==torch.uint8 for value in rng.values())
            rng_count+=1
    assert final_receipt['budget']['new_teacher_queries']==0
    assert final_receipt['budget']['teacher_queries']==json.loads((args.stage1_run/'training-result.json').read_bytes())['budget']['teacher_queries']
    assert final_receipt['budget']['admitted_learner_microsteps']<=100000
    assert final_receipt['budget']['wall_seconds']<=7200
    receipt={'schema':'86-tcg-agent/actorcritic-independent-audit/1','status':'passed',
             'bindings_sha256':sha(bindings_path),'source_files_verified':len(bindings['artifacts']),
             'training_result_sha256':sha(ROOT/'actorcritic-result.json'),'checkpoint_identities':checkpoint_identities,
             'stage3_actor_parameter_tensors_unchanged':len(actor_names),'stage3_critic_parameter_tensors_changed':len(critic_changed),
             'stage4_actor_parameter_tensors_changed':len(stage4_actor_changed),'static_buffers_unchanged':buffer_count,
             'calibration_recomputed':calibration,'all_128_heldout_actor_logits_exact':True,'ppo_independent_targets':ppo,
             'episode_statuses':statuses,'episode_rng_files_verified':rng_count,'native_counters':dict(native_counters),
             'budget':final_receipt['budget'],'new_teacher_queries':0,'no_games_started':True,'no_optimizer_updates':True,
             'limitations':['Signed values and old action probabilities were recomputed on stored actor frames, without native replay.',
                            'Terminal outcomes use actual recorded native receipts; engine rule parity is not established.',
                            'Zero new teacher queries is supported by source closure and runner receipts, not process-call instrumentation.',
                            'Calibration and PPO pipeline correctness do not establish playing strength.']}

    receipt.update(tool_identity())
    return receipt


def audit_evaluation(args):
    ROOT = args.evaluation_run
    TRAIN = args.stage1_run
    AC = args.actorcritic_run
    PRIVATE = args.output
    def read(path):
        return json.loads(Path(path).read_bytes())
    def seed(seed, domain, *identity):
        return int.from_bytes(hashlib.sha256(canonical([seed, domain, *identity])).digest()[:4], 'little') or 1
    assert (ROOT/'evaluation-result.json').is_file()
    manifest=read(ROOT/'manifest.json');result=read(ROOT/'evaluation-result.json')
    bindings=read(args.continuation_bindings)
    for path,expected in bindings['artifacts'].items():assert sha(path)==expected,('source changed',path)
    assert hashlib.sha256(canonical(bindings['artifacts'])).hexdigest()==bindings['artifact_manifest_sha256']==manifest['artifact_manifest_sha256']
    assert sha(TRAIN/'training-result.json')==manifest['stage1_result_sha256']
    assert sha(AC/'actorcritic-result.json')==manifest['actorcritic_result_sha256']
    assert result['teacher_off'] and result['teacher_model_loads']==0 and result['new_teacher_queries']==0 and result['frozen_players_unchanged']
    assert manifest['teacher_access']=='forbidden'
    assert hashlib.sha256(canonical(manifest['protocol'])).hexdigest()==manifest['protocol_sha256']
    expected_players={'initial':(TRAIN/'initial.pt',read(TRAIN/'initial.json'),'model_identity'),
                      'pre_rl':(AC/'stage3.pt',read(AC/'stage3.json'),'identity'),
                      'post_rl':(AC/'stage4.pt',read(AC/'stage4.json'),'identity')}
    checkpoint_hashes={}
    for arm,(path,receipt,key) in expected_players.items():
        assert sha(path)==receipt['weights_sha256']
        assert manifest['players'][arm]['state_digest']==receipt[key]['state_digest']
        assert manifest['players'][arm]['config']==receipt[key]['config']
        checkpoint_hashes[arm]=sha(path)
    plan=read(ROOT/'evaluation-plan.json')['assignments']
    assert plan==read(TRAIN/'evaluation-plan.json')['assignments']
    derived=[]
    evaluation_seed=manifest['protocol']['seeds']['evaluation']
    for left in range(6):
        for right in range(6):
            for seat in (0,1):
                derived.append({'index':len(derived),'learner_deck':left,'opponent_deck':right,'learner_seat':seat,
                                'engine_seed':seed(evaluation_seed,'stage1-eval-engine-v1',left,right,seat),
                                'learner_seed':seed(evaluation_seed,'stage1-eval-learner-v1',left,right,seat),
                                'opponent_seed':seed(evaluation_seed,'stage1-eval-opponent-v1',left,right,seat)})
    assert derived==plan and len({item['engine_seed'] for item in plan})==72
    games=[json.loads(line) for line in (ROOT/'evaluation-games.jsonl').read_bytes().splitlines()]
    indexed={(row['arm'],row['index']):row for row in games}
    assert len(games)==len(indexed)==216
    counts={};arm_bounds={};native_counts={}
    prior_budgets={'initial':{'wall_seconds':0,'admitted_learner_microsteps':0},
                   'pre_rl':read(AC/'stage3.json')['budget'],'post_rl':read(AC/'stage4.json')['budget']}
    for arm in ('initial','pre_rl','post_rl'):
        selected=[]
        for assignment in plan:
            row=indexed[arm,assignment['index']]
            assert all(row[key]==value for key,value in assignment.items())
            assert row['status'] in ('terminal','censored','fault','not_started')
            if row['status']=='terminal':
                assert row['winner'] in (0,1,2)
                score=0.5 if row['winner']==2 else float(row['winner']==row['learner_seat'])
                assert row['score']==score
                assert row['native_stats']['gamesFinished']==1
            else:assert row['score'] is None
            selected.append(row)
        terminal=[row for row in selected if row['status']=='terminal']
        complete_score=sum(row['score'] for row in terminal)
        arm_bounds[arm]=[complete_score/72,(complete_score+72-len(terminal))/72]
        charged=sum(sum(choice['learner'] for choice in row.get('choices',[])) for row in selected)
        budget=result['arm_budgets'][arm]
        assert budget['admitted_learner_microsteps']==prior_budgets[arm]['admitted_learner_microsteps']+charged
        assert prior_budgets[arm]['wall_seconds']<=budget['wall_seconds']<=7200
        assert budget['admitted_learner_microsteps']<=100000 and budget['evaluation_wall_seconds']<=1800
        assert budget['teacher_queries']==(0 if arm=='initial' else prior_budgets[arm]['teacher_queries'])
        counts[arm]={'statuses':dict(Counter(row['status'] for row in selected)),'wins':sum(row['score']==1 for row in terminal),
                     'draws':sum(row['score']==0.5 for row in terminal),'losses':sum(row['score']==0 for row in terminal),
                     'score_bounds':arm_bounds[arm],'explicit_learner_choices':charged,
                     'genuine_choices':sum(row.get('learner_genuine_choices',0) for row in selected)}
        stats=Counter()
        for row in selected:stats.update(row.get('native_stats',{}))
        native_counts[arm]=dict(stats)

    alpha=manifest['protocol']['evaluation']['per_contrast_alpha']
    assert alpha==0.025
    radius=math.sqrt(2*math.log(2/alpha)/36)
    contrasts={}
    for name,candidate,control in [('policy_transfer','pre_rl','initial'),('ppo_improvement','post_rl','pre_rl')]:
        cluster_bounds=[]
        for left in range(6):
            for right in range(6):
                lo=hi=0.0
                for seat in (0,1):
                    index=(left*6+right)*2+seat
                    a,b=indexed[candidate,index],indexed[control,index]
                    alo=ahi=a['score'] if a['status']=='terminal' else None
                    blo=bhi=b['score'] if b['status']=='terminal' else None
                    if alo is None:alo,ahi=0.0,1.0
                    if blo is None:blo,bhi=0.0,1.0
                    lo+=(alo-bhi)/2;hi+=(ahi-blo)/2
                cluster_bounds.append((lo,hi))
        difference=[sum(item[0] for item in cluster_bounds)/36,sum(item[1] for item in cluster_bounds)/36]
        assert abs(difference[0]-(arm_bounds[candidate][0]-arm_bounds[control][1]))<1e-12
        assert abs(difference[1]-(arm_bounds[candidate][1]-arm_bounds[control][0]))<1e-12
        interval=[max(-1,difference[0]-radius),min(1,difference[1]+radius)]
        verdict='improved_against_uniform_anchor' if interval[0]>0 else ('worse_against_uniform_anchor' if interval[1]<0 else 'inconclusive')
        contrasts[name]={'candidate':candidate,'control':control,'clusters':36,'difference_bounds':difference,
                         'interval':interval,'verdict':verdict}
    receipt={'schema':'86-tcg-agent/native-evaluation-independent-audit/1','status':'passed',
             'source_files_verified':len(bindings['artifacts']),'manifest_sha256':sha(ROOT/'manifest.json'),
             'evaluation_plan_sha256':sha(ROOT/'evaluation-plan.json'),'evaluation_games_sha256':sha(ROOT/'evaluation-games.jsonl'),
             'evaluation_result_sha256':sha(ROOT/'evaluation-result.json'),'checkpoint_sha256':checkpoint_hashes,
             'assignments_verified':216,'independent_schedule_derivation_exact':True,'outcome_perspective_verified':True,
             'summary':counts,'native_counts':native_counts,'contrasts':contrasts,'arm_budgets':result['arm_budgets'],
             'family_alpha':0.05,'per_contrast_alpha':alpha,'hoeffding_difference_radius':radius,
             'teacher_loads_recorded':0,'new_teacher_queries_recorded':0,
             'limitations':['Checkpoint files and recorded model identities were checked without loading models in this audit.',
                            'Unchanged in-memory players and absent teacher queries rely on bound source plus runner attestations, not independent process instrumentation.',
                            'No native game replay or omitted-wire projection reconstruction was performed.',
                            'Outcomes concern one student seed and a weak uniform legal opponent in six historical decks.'],
             'no_models_loaded':True,'no_games_started':True,'no_optimizer_updates':True}

    receipt.update(tool_identity())
    return receipt


def main():
    if sys.flags.optimize:
        raise RuntimeError('Audit assertions require Python without -O or -OO')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage1-run", "actorcritic-run", "evaluation-run", "stage1-bindings", "continuation-bindings", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if key != "output":
            setattr(args, key, value.resolve(strict=True))
    if any(args.output.resolve().is_relative_to(run) for run in (args.stage1_run, args.actorcritic_run, args.evaluation_run)):
        raise ValueError('Audit output must be outside the immutable experiment directories')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if os.name != "posix" or not hasattr(os, "nice"):
            raise RuntimeError("Use the pinned Linux or WSL CPU runtime")
        os.nice(19 - os.nice(0))
        os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        global torch, native
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tcg import native_policy as native
        bindings = json.loads(args.stage1_bindings.read_bytes())
        continuation = json.loads(args.continuation_bindings.read_bytes())
        if Path(bindings["mew_root"]).resolve() != Path(continuation["mew_root"]).resolve():
            raise ValueError("The stages must use the same pinned private Mew tree")
        os.environ["FASTENC_LIB"] = bindings["engine_library"]
        native.configure_mew(bindings["mew_root"])
        receipts = (("stage1-audit.json", audit_stage1), ("probe-novelty.json", audit_probe_novelty),
                    ("actorcritic-audit.json", audit_actorcritic), ("evaluation-audit.json", audit_evaluation))
        for name, audit in receipts:
            result = audit(args)
            with (args.output / name).open("x") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            print(json.dumps({"receipt": name, "status": result["status"], "sha256": file_sha(args.output / name)}), flush=True)
        for path, expected in continuation["artifacts"].items():
            if file_sha(path) != expected:
                raise ValueError("A bound source changed during the audit")
    except Exception as error:
        with (args.output / "failure.json").open("x") as stream:
            json.dump({"status": "failed", "error": f"{type(error).__name__}: {error}",
                       "audit_tool_version": VERSION, "audit_tool_sha256": file_sha(__file__)}, stream, indent=2)
        raise


if __name__ == "__main__":
    main()
