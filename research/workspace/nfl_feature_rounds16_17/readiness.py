"""Independent readiness per new round; pending review is not an execution failure."""
from __future__ import annotations
from audit_io import digest,hash_json,safe_file,read_json,atomic_json


def require_tests(a):
    from bridge import code_signature
    path=a.out.parent/'nfl-feature-round16-results'/'tests_receipt.json'
    r=read_json(safe_file(path.parent,path.name))
    if (r.get('status')!='tests_passed' or r.get('source_signature')!=code_signature(a.kit)
        or r.get('tests_run',0)<=0 or any(r.get(k,1) for k in ('failures','errors','skipped'))):
        raise ValueError('Current user-executed tests are required for this new source')
    return r


def labels(a):
    contract=read_json(a.kit/'input_contract.json')
    path=a.out.parent/'nfl-feature-round14-results'/'label_readiness.json'
    value=read_json(safe_file(path.parent,path.name))
    if (hash_json(value)!=contract['shared_label_receipt_sha256']
        or value.get('source_signature')!=contract['shared_label_source_signature']
        or value.get('status')!='training_labels_complete'):
        raise ValueError('Reviewed shared training-label receipt changed')
    return value


def reviewed_readiness(a):
    from bridge import code_signature
    require_tests(a);labels(a)
    pre=read_json(safe_file(a.out,'preflight.json'));smoke=read_json(safe_file(a.out,'smoke.json'))
    if (pre.get('round')!=a.round_no or smoke.get('round')!=a.round_no
        or smoke.get('status')!='family_smoke_passed' or pre['signature']!=smoke['signature']):
        raise ValueError('The current round must have a matching preflight and 32-play smoke')
    hashes={'tests':digest(a.out.parent/'nfl-feature-round16-results'/'tests_receipt.json'),
            'labels':digest(a.out.parent/'nfl-feature-round14-results'/'label_readiness.json'),
            'preflight':digest(a.out/'preflight.json'),'smoke':digest(a.out/'smoke.json')}
    return {'source_signature':code_signature(a.kit),'readiness_hash':hash_json(hashes),'evidence':hashes}


def require_review_release(a):
    ready=reviewed_readiness(a)
    path=a.out/'review_release.json'
    if not path.exists():
        raise ValueError('REVIEW PENDING: new 16/17 fits are held until completed 14/15 results and '
                         'the larger-training-data decision have been reviewed. No stage was launched.')
    r=read_json(safe_file(a.out,'review_release.json'))
    if (r.get('decision')!='bounded_exploratory_study_after_readiness_review'
        or r.get('round')!=a.round_no or r.get('max_scientific_models')!=3
        or r.get('source_signature')!=ready['source_signature']
        or r.get('readiness_hash')!=ready['readiness_hash']
        or r.get('acknowledge_prior_gates_failed') is not True
        or r.get('acknowledge_reused_evaluation') is not True
        or not isinstance(r.get('reason'),str) or len(r['reason'].strip())<30
        or not isinstance(r.get('data_scale_decision'),str) or len(r['data_scale_decision'].strip())<30):
        raise ValueError('Release is incomplete or does not match current readiness')
    return r


def status(a):
    try:
        r=require_review_release(a)
        return {'status':'ready','ready':True,'release':r}
    except (FileNotFoundError,ValueError) as exc:
        return {'status':'review_pending','ready':False,'message':str(exc),
                'new_model_fits':0,'instruction':'Return readiness evidence; do not fabricate a release.'}
