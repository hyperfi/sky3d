#!/usr/bin/env python3
"""Compile retained report source/assets using an existing compiler; no simulations."""
import argparse
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',help='Existing pdflatex executable; auto-detected by default')
    args=parser.parse_args()
    compiler=args.compiler or shutil.which('pdflatex') or shutil.which('pdflatex.exe')
    if not compiler:
        raise SystemExit('No existing PDFLaTeX found. Supply --compiler; this script does not install TeX.')
    build=HERE/'build';build.mkdir(exist_ok=True)
    windows=compiler.lower().endswith('.exe')
    outarg=str(build)
    if windows:
        outarg=subprocess.check_output(['wslpath','-w',str(build)],text=True).strip()
    command=[compiler,'-interaction=nonstopmode','-halt-on-error',f'-output-directory={outarg}',
             'sky3d_single_gpu_report.tex']
    if windows: command.insert(1,'--disable-installer')
    for pass_number in range(1,4):
        result=subprocess.run(command,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (build/f'compile-pass-{pass_number}.txt').write_bytes(result.stdout)
        if result.returncode:
            print(result.stdout.decode(errors='replace')[-6500:])
            raise SystemExit(f'PDFLaTeX failed on pass {pass_number}; inspect {build}')
    output=REPO/'output/pdf';output.mkdir(parents=True,exist_ok=True)
    target=output/'sky3d_single_gpu_report.pdf'
    shutil.copy2(build/'sky3d_single_gpu_report.pdf',target)
    print('Compiled report:',target)
    print('Compiler:',compiler)

if __name__=='__main__':main()
