"""Explicit scientific-review checkpoint. Preparation cannot authorize fitting."""
from __future__ import annotations
from audit_io import digest, hash_json, safe_file, read_json


def reviewed_readiness(a):
    from bridge import code_signature
    source = code_signature(a.kit)
    home = a.out.parent
    root14 = home/'nfl-feature-round14-results'
    root15 = home/'nfl-feature-round15-results'
    tests = read_json(safe_file(root14,'tests_receipt.json'))
    if (tests.get('status')!='tests_passed' or tests.get('source_signature')!=source
        or tests.get('tests_run',0)<=0 or tests.get('failures',1)!=0
        or tests.get('errors',1)!=0 or tests.get('skipped',1)!=0):
        raise ValueError('Current user-executed tests are required')
    label = read_json(safe_file(root14,'label_readiness.json'))
    if label.get('status')!='training_labels_complete' or label.get('source_signature')!=source:
        raise ValueError('Training-label coverage requires review before fitting')
    evidence={'tests':digest(root14/'tests_receipt.json'), 'labels':digest(root14/'label_readiness.json')}
    for r,folder in ((14,root14),(15,root15)):
        smoke=read_json(safe_file(folder,'smoke.json'))
        pre=read_json(safe_file(folder,'preflight.json'))
        if smoke.get('status')!='family_smoke_passed' or smoke['signature']!=pre['signature']:
            raise ValueError('Both family smoke tests must pass')
        evidence[f'round{r}_smoke']=digest(folder/'smoke.json')
        evidence[f'round{r}_preflight']=digest(folder/'preflight.json')
    return {'source_signature':source,'readiness_hash':hash_json(evidence),'evidence':evidence}


def require_review_release(a):
    """A manually authored review decision is required after the first report.

    The package deliberately includes a TEMPLATE, not an active release. Passing
    software checks does not silently overturn the old failed scientific gates.
    """
    ready=reviewed_readiness(a)
    p=a.out/'review_release.json'
    if not p.is_file():
        raise ValueError('STOP FOR REVIEW: return both readiness reports before profiling or fitting. '
                         'No active review_release.json was supplied with this package.')
    release=read_json(safe_file(a.out,'review_release.json'))
    if (release.get('decision')!='bounded_exploratory_study_after_readiness_review'
        or release.get('readiness_hash')!=ready['readiness_hash']
        or release.get('source_signature')!=ready['source_signature']
        or release.get('round')!=a.round_no
        or release.get('max_scientific_models')!=3
        or release.get('acknowledge_prior_gates_failed') is not True
        or release.get('acknowledge_reused_evaluation') is not True
        or not isinstance(release.get('reason'),str) or len(release['reason'].strip())<30
        or not isinstance(release.get('data_scale_decision'),str) or len(release['data_scale_decision'].strip())<30):
        raise ValueError('Review decision is incomplete or does not match current readiness')
    return release
