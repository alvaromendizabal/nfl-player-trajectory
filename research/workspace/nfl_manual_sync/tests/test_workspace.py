import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('nfl_workspace', ROOT / 'nfl_workspace.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


@pytest.fixture
def run(tmp_path):
    r = w.Run(tmp_path / 'receipts', 'test', 60)
    yield r
    r.closed.set()


def g(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.STDOUT, text=True).strip()


@pytest.fixture
def repo(tmp_path):
    p = tmp_path / 'repo'; p.mkdir()
    g(p, 'init', '-b', 'main')
    g(p, 'config', 'user.email', 'fixture@example.invalid')
    g(p, 'config', 'user.name', 'Local test fixture')
    g(p, 'remote', 'add', 'origin', 'https://github.com/' + w.REPOSITORY + '.git')
    (p / '.gitignore').write_text('data/\nartifacts/\n.venv/\nignored.txt\n')
    (p / 'source.py').write_text('VALUE = 1\n')
    g(p, 'add', '.'); g(p, 'commit', '-m', 'base')
    base = g(p, 'rev-parse', 'HEAD')
    g(p, 'checkout', '-b', 'upstream')
    (p / 'source.py').write_text('VALUE = 2\n')
    (p / 'new.py').write_text('NEW = True\n')
    g(p, 'add', '.'); g(p, 'commit', '-m', 'advance')
    target = g(p, 'rev-parse', 'HEAD')
    g(p, 'update-ref', 'refs/remotes/origin/main', target)
    g(p, 'checkout', 'main')
    return p, base, target


def plan(repo, run):
    p, base, _ = repo
    return w.prepare_sync(p, run, fetch=False, minimum=base,
                          check_quality=lambda sha: {'conclusion':'success', 'head_sha':sha})


def test_clean_fast_forward_preserves_raw_and_artifacts(repo, run, monkeypatch):
    p, base, target = repo
    (p / 'data/raw').mkdir(parents=True)
    (p / 'data/raw/input.csv').write_bytes(b'original raw bytes\n')
    (p / 'artifacts').mkdir(); (p / 'artifacts/model.bin').write_bytes(b'private model')
    (p / '.venv').mkdir(); (p / '.venv/keep').write_text('existing environment')
    (p / 'untracked_receipt.json').write_text('{}')
    before = {x: x.read_bytes() for x in [p/'data/raw/input.csv', p/'artifacts/model.bin', p/'.venv/keep', p/'untracked_receipt.json']}
    original = w.git
    def local_git(root, r, *args, **kwargs):
        if args[:1] == ('ls-remote',):
            return target + '\trefs/heads/main'
        return original(root, r, *args, **kwargs)
    monkeypatch.setattr(w, 'git', local_git)
    result = w.apply_sync(p, run, plan(repo, run))
    assert result['status'] == 'source_synced_data_preserved'
    assert g(p, 'rev-parse', 'HEAD') == target
    assert all(x.read_bytes() == v for x,v in before.items())
    assert result['raw_files_sha256_verified_unchanged'] == 1
    assert Path(result['git_reference_backup']).is_file()
    g(p, 'bundle', 'verify', result['git_reference_backup'])


def test_merged_feature_branch_moves_to_main(repo, run, monkeypatch):
    p, _, target = repo
    g(p, 'checkout', '-b', 'old-feature')
    original = w.git
    monkeypatch.setattr(w, 'git', lambda root,r,*a,**k: target+'\trefs/heads/main' if a[:1]==('ls-remote',) else original(root,r,*a,**k))
    w.apply_sync(p, run, plan(repo, run))
    assert g(p,'branch','--show-current') == 'main'
    assert g(p,'rev-parse','HEAD') == target


def test_dirty_tracked_preserved(repo, run):
    p, base, _ = repo
    (p/'source.py').write_text('OWNER EDIT\n')
    with pytest.raises(ValueError, match='Tracked edits'):
        plan(repo,run)
    assert (p/'source.py').read_text() == 'OWNER EDIT\n'
    assert g(p,'rev-parse','HEAD') == base


def test_staged_edits_preserved(repo, run):
    p,_,_ = repo; (p/'source.py').write_text('staged'); g(p,'add','source.py')
    with pytest.raises(ValueError,match='Tracked edits'): plan(repo,run)
    assert g(p,'diff','--cached','--name-only') == 'source.py'


def test_divergent_head_refused(repo, run):
    p,_,_ = repo; (p/'local.py').write_text('mine'); g(p,'add','.'); g(p,'commit','-m','local')
    head=g(p,'rev-parse','HEAD')
    with pytest.raises(ValueError,match='not in main'): plan(repo,run)
    assert g(p,'rev-parse','HEAD') == head


def test_untracked_collision_refused(repo, run):
    p,_,_=repo; (p/'new.py').write_text('owner untracked')
    with pytest.raises(ValueError,match='collides'): plan(repo,run)
    assert (p/'new.py').read_text()=='owner untracked'


def test_ignored_collision_refused(repo, run):
    p,base,_=repo; g(p,'checkout','upstream'); (p/'ignored.txt').write_text('upstream')
    g(p,'add','-f','ignored.txt'); g(p,'commit','-m','collision'); target=g(p,'rev-parse','HEAD')
    g(p,'update-ref','refs/remotes/origin/main',target); g(p,'checkout','main')
    (p/'ignored.txt').write_text('owner ignored')
    with pytest.raises(ValueError,match='collides'): plan((p,base,target),run)
    assert (p/'ignored.txt').read_text()=='owner ignored'


def test_protected_incoming_refused(repo,run):
    p,base,_=repo; g(p,'checkout','upstream'); (p/'data').mkdir(); (p/'data/new.csv').write_text('bad')
    g(p,'add','-f','data/new.csv'); g(p,'commit','-m','bad public data'); target=g(p,'rev-parse','HEAD')
    g(p,'update-ref','refs/remotes/origin/main',target); g(p,'checkout','main')
    with pytest.raises(ValueError,match='protected'): plan((p,base,target),run)


def test_quality_failure_keeps_checkout(repo,run):
    p,base,_=repo
    def fail(sha): raise ValueError('quality not passed')
    with pytest.raises(ValueError,match='quality'):
        w.prepare_sync(p,run,fetch=False,minimum=base,check_quality=fail)
    assert g(p,'rev-parse','HEAD')==base


def test_wrong_origin_refused(repo,run):
    p,_,_=repo; g(p,'remote','set-url','origin','https://github.com/other/repo.git')
    with pytest.raises(ValueError,match='Origin'): plan(repo,run)


@pytest.mark.parametrize('name',['../secret','/tmp/escape','a/../../escape','a\\b','C:/b','a/./b',''])
def test_unsafe_path(name,tmp_path):
    with pytest.raises(ValueError): w.safe_destination(tmp_path,name)


def test_symlink_parent_refused(tmp_path):
    (tmp_path/'outside').mkdir(); (tmp_path/'link').symlink_to(tmp_path/'outside',target_is_directory=True)
    with pytest.raises(ValueError,match='Symlinks'): w.safe_destination(tmp_path,'link/file')


def item(data=b'fixture',path='data/raw/test.csv'):
    return {'path':path,'sha256':hashlib.sha256(data).hexdigest(),'size':len(data)}


def test_atomic_restore_and_resume(tmp_path,run):
    content=b'fixture'; record=item(content)
    assert w.install_verified(tmp_path,record,lambda:iter([content]),run)=='restored'
    def forbidden(): raise AssertionError('must not re-download')
    assert w.install_verified(tmp_path,record,forbidden,run)=='reused'
    assert (tmp_path/record['path']).read_bytes()==content


def test_divergent_restore_never_overwrites(tmp_path,run):
    target=tmp_path/'data/raw/test.csv'; target.parent.mkdir(parents=True); target.write_bytes(b'owner')
    with pytest.raises(ValueError,match='differs'):
        w.install_verified(tmp_path,item(),lambda:iter([b'fixture']),run)
    assert target.read_bytes()==b'owner'


@pytest.mark.parametrize('blocks',[[b'wrong!!'],[b'fix'],[b'fixtureEXTRA']])
def test_corrupt_or_partial_download_not_installed(tmp_path,run,blocks):
    with pytest.raises(ValueError): w.install_verified(tmp_path,item(),lambda:iter(blocks),run)
    assert not (tmp_path/'data/raw/test.csv').exists()
    assert not list((tmp_path/'data/raw').glob('.nfl-input-*'))


def test_manifest_hash_and_duplicate_guard():
    payload=json.dumps({'format':1,'files':[item()]}).encode()
    key='snapshots/'+hashlib.sha256(payload).hexdigest()+'.json'
    assert w.validate_manifest(payload,key)==[item()]
    with pytest.raises(ValueError,match='SHA256'): w.validate_manifest(payload+b' ',key)
    duplicate=json.dumps({'format':1,'files':[item(),item()]}).encode()
    with pytest.raises(ValueError,match='duplicate'):
        w.validate_manifest(duplicate,'snapshots/'+hashlib.sha256(duplicate).hexdigest()+'.json')


def test_select_only_raw_and_exact_cache():
    cache={'path':'artifacts/temporal/samples.pkl','sha256':w.SAMPLE_SHA,'size':10}
    assert w.select_inputs([[item(),cache,item(path='artifacts/other-model.pkl')]])==[cache,item()]


def test_snapshot_conflict_fails():
    with pytest.raises(ValueError,match='disagree'):
        w.select_inputs([[item(b'one')],[item(b'two')]])


def test_raw_audit_complete_and_missing(tmp_path,run):
    train=tmp_path/'data/raw/train'; train.mkdir(parents=True)
    for week in range(1,19):
        for prefix,fields in [('input',w.REQUIRED_INPUT),('output',w.REQUIRED_OUTPUT)]:
            columns=sorted(fields)
            (train/f'{prefix}_2023_w{week:02d}.csv').write_text(','.join(columns)+'\n'+','.join(['1']*len(columns))+'\n')
    raw=train.parent
    (raw/'test.csv').write_text('game_id\n1\n'); (raw/'test_input.csv').write_text('game_id\n1\n')
    api=raw/'kaggle_evaluation'; api.mkdir()
    for n in range(11): (api/f'file{n}.py').write_text('# organizer fixture\n')
    result=w.data_audit(tmp_path,run)
    assert result['status']=='raw_readiness_passed' and result['file_count']==49
    (train/'output_2023_w18.csv').unlink()
    result=w.data_audit(tmp_path,run)
    assert result['status']=='raw_readiness_incomplete'
    assert any('output_2023_w18.csv' in x for x in result['problems'])


def test_raw_symlink_refused(tmp_path,run):
    (tmp_path/'data').mkdir(); (tmp_path/'data/raw').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'): w.inventory_raw(tmp_path,run)


def test_command_failure_and_timeout(tmp_path,run):
    with pytest.raises(RuntimeError):
        w.command([sys.executable,'-c','raise SystemExit(3)'],tmp_path,run,'failed',3)
    with pytest.raises(TimeoutError):
        w.command([sys.executable,'-c','import time; time.sleep(10)'],tmp_path,run,'timeout',0.1)


def test_atomic_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError): w.atomic_json(tmp_path/'nan.json',{'x':float('nan')})
    assert not (tmp_path/'nan.json').exists()


def test_unreviewed_ancestry_refused(repo,run):
    p,base,target=repo
    with pytest.raises(ValueError,match='reviewed baseline'):
        w.prepare_sync(p,run,fetch=False,minimum='f'*40,check_quality=lambda s:{})
    assert g(p,'rev-parse','HEAD')==base


def test_local_main_divergence_refused(repo,run):
    p,base,target=repo
    (p/'owner.txt').write_text('local-main'); g(p,'add','.'); g(p,'commit','-m','unmerged local-main')
    g(p,'checkout','-b','old-feature',base)
    with pytest.raises(ValueError,match='Local main diverges'):
        plan(repo,run)
    assert g(p,'branch','--show-current')=='old-feature'


def test_different_raw_file_does_not_get_unlinked(tmp_path,run):
    p=tmp_path/'data/raw/test.csv'; p.parent.mkdir(parents=True); p.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='differs'):
        w.install_verified(tmp_path,item(b'xxxxxxx'),lambda:iter([b'xxxxxxx']),run)
    assert p.read_bytes()==b'fixture'


def test_restore_concurrent_writer_preserved(tmp_path,run):
    target=tmp_path/'data/raw/test.csv'
    def blocks():
        target.write_bytes(b'concurrent owner')
        yield b'fixture'
    with pytest.raises(FileExistsError):
        w.install_verified(tmp_path,item(),blocks,run)
    assert target.read_bytes()==b'concurrent owner'


def test_git_gets_no_destructive_flags():
    text=(ROOT/'nfl_workspace.py').read_text()
    assert '"--hard"' not in text
    assert '["git", "clean"' not in text
    assert '"--force"' not in text


def test_python39_syntax():
    import ast
    ast.parse((ROOT/'nfl_workspace.py').read_text(),feature_version=(3,9))


def test_return_report_excludes_raw_and_logs(tmp_path,run):
    import zipfile
    w.atomic_json(run.home/'latest_plan.json',{'status':'fixture'})
    (run.home/'private.csv').write_text('do not share')
    (run.home/'secret.log').write_text('do not share')
    result=w.return_report(run)
    with zipfile.ZipFile(result['path']) as archive:
        assert sorted(archive.namelist())==['CONTENTS.txt','latest_plan.json']
