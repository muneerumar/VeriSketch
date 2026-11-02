"""Check the content boundary of the release snapshot (no training libraries needed)."""
from pathlib import Path
import argparse,json,re
from release_results import BLOCKED

def inspect_json(o,path,issues):
    if isinstance(o,dict):
        for k,v in o.items():
            if k in BLOCKED or k.endswith('_per_client_mae'):issues.append(f'{path}: forbidden record/client field {k}')
            inspect_json(v,path,issues)
    elif isinstance(o,list):
        for v in o:inspect_json(v,path,issues)

def check(root):
    root=Path(root);issues=[];files=0
    forbidden_suffix={'.pdf','.tex','.bib','.bbl','.docx','.doc','.pt','.pth','.pkl','.pickle','.safetensors','.npy','.npz','.zip','.log'}
    forbidden_dirs={'paper','papers','manuscript','manuscripts','logs','model','features','private','reviews','.git'}
    secret_patterns=[r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',r'\bAKIA[A-Z0-9]{16}\b']
    for p in root.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or '.git' in p.relative_to(root).parts:continue
        rel=p.relative_to(root);files+=1
        if set(rel.parts[:-1])&forbidden_dirs or p.suffix.lower() in forbidden_suffix:issues.append(f'{rel}: private/generated artifact')
        if rel.parts[0]=='results':issues.append(f'{rel}: local outputs must not be staged')
        if p.name.startswith('.env'):issues.append(f'{rel}: environment file')
        if any(s in p.name.lower() for s in ['private_audit','response_to_review','reviewer_response','manuscript']):issues.append(f'{rel}: private manuscript/review material')
        if p.name in {'split_manifest.csv','dataset.json','post_id_divisions.json'} or p.name.endswith('_xai.csv'):issues.append(f'{rel}: row-level source or evaluation data')
        if p.name in {'build_latex.py','paper_values.py','paper_figures.py','main.tmpl.tex'}:issues.append(f'{rel}: manuscript generation source')
        text=p.read_text(errors='replace')
        if re.search(r'/(?:Users|home|Volumes)/[^\s\"\']+',text):issues.append(f'{rel}: absolute user-machine path')
        if any(re.search(pattern,text) for pattern in secret_patterns):issues.append(f'{rel}: credential-like value')
        if rel.parts[0]=='public_results' and p.suffix=='.json':
            try:inspect_json(json.loads(text),str(rel),issues)
            except json.JSONDecodeError:issues.append(f'{rel}: invalid JSON')
        if p.suffix=='.csv' and rel.parts[0]=='public_results':
            import csv,io
            headers=next(csv.reader(io.StringIO(text)),[])
            if any(k in BLOCKED for k in headers):issues.append(f'{rel}: record/client CSV columns')
    return files,issues

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',nargs='?',default='.')
    a=p.parse_args();n,issues=check(a.path)
    for issue in issues:print(issue)
    if issues:raise SystemExit(f'Release check failed: {len(issues)} findings')
    print(f'Release content check passed ({n} files). This is not a security or privacy proof.')

if __name__=='__main__':main()
