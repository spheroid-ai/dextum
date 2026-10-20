"""One registered source-metadata migration; full training payload is unchanged."""
import argparse,gzip,hashlib,json,os,pickle,time
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Trusted checkpoint operations on Vast only')
import vendor.aero_official.vision_env  # bootstrap the pinned playground types used by the pickle


def digest_payload(payload):
    class Sink:
        def __init__(self):self.hash=hashlib.sha256();self.bytes=0
        def write(self,data):
            view=memoryview(data);self.hash.update(view);self.bytes+=view.nbytes
            return view.nbytes
    sink=Sink();pickle.Pickler(sink,protocol=5).dump(payload)
    return dict(sha256=sink.hash.hexdigest(),bytes=sink.bytes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--old-release',type=Path,required=True);p.add_argument('--new-release',type=Path,required=True)
    a=p.parse_args();case=a.run/'cases/c_touch_ideal/c_touch'
    assert json.loads((a.run/'CLUSTERED_GUARD_AUDIT.json').read_text())['cases']['c_touch_ideal']['state']=='PASS'
    old=case/'full_state/latest.pkl.gz';new=case/'full_state/guard_migrated_7208960.pkl.gz'
    assert not new.exists()
    with gzip.open(old,'rb') as f:record=pickle.load(f)
    assert record['version']==1
    before=digest_payload(record['payload']);metadata=record['metadata']
    original_metadata=json.loads(json.dumps(metadata));changes={}
    allowed={'pass1c/criteria.py','pass1c/compare.py'}
    for source,old_hash in metadata['sources'].items():
        assert hashlib.sha256((a.old_release/source).read_bytes()).hexdigest()==old_hash,source
        new_hash=hashlib.sha256((a.new_release/source).read_bytes()).hexdigest()
        if new_hash!=old_hash:
            assert source in allowed,source
            changes[source]=dict(before=old_hash,after=new_hash)
            metadata['sources'][source]=new_hash
    assert set(changes)==allowed
    ts=record['payload']['carry'][0]
    assert int(ts.env_steps.to_numpy()[0])==7208960
    assert 'pending' in record['payload']
    tmp=new.with_suffix('.tmp')
    with gzip.open(tmp,'wb',compresslevel=1) as f:pickle.dump(record,f,protocol=5)
    os.replace(tmp,new)
    del record
    with gzip.open(new,'rb') as f:restored=pickle.load(f)
    after=digest_payload(restored['payload']);assert after==before
    report=dict(state='PASS',authority='User delegated guard ruling',created_at=time.time(),
                source=str(old),destination=str(new),optimizer_steps=7208960,
                metadata_before=original_metadata,metadata_after=restored['metadata'],
                changed_sources=changes,payload_before=before,payload_after=after,
                pending_batch_retained=True,new_training_attempt=False)
    (a.run/'GUARD_CHECKPOINT_MIGRATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.startswith('metadata')}),flush=True)

if __name__=='__main__':main()
