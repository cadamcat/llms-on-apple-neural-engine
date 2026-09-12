"""Small path, identity and JSON helpers; no device imports."""
from pathlib import Path
import hashlib,json,subprocess,sys,importlib.metadata as metadata,platform
PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[1]

def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def file_hashes(path):return {p.relative_to(path).as_posix():sha(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}
def environment():
    packages={}
    for name in ['numpy','torch','coreai-core','coreai-torch','coremltools']:
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    result={'python':sys.version.split()[0],'architecture':platform.machine(),'system':platform.system(),'packages':packages}
    if sys.platform=='darwin':
        for name,cmd in [('os',['sw_vers']),('xcode',['xcodebuild','-version']),('sdk',['xcrun','--sdk','macosx','--show-sdk-version']),('chip',['sysctl','-n','machdep.cpu.brand_string']),('memory',['sysctl','-n','hw.memsize'])]:
            p=subprocess.run(cmd,text=True,capture_output=True,timeout=20)
            result[name]={'returncode':p.returncode,'value':p.stdout.strip(),'error':p.stderr.strip()[:1000]}
    return result

def source_identity():
    # Exact working-tree identity before a user creates the first commit.
    paths=list(PACKAGE.rglob('*.py'))+list((PACKAGE/'native').glob('*.swift'))
    result={p.relative_to(PACKAGE).as_posix():sha(p) for p in sorted(paths)}
    for name in ['pyproject.toml','uv.lock']:
        if (ROOT/name).is_file():result['../../'+name]=sha(ROOT/name)
    return result
