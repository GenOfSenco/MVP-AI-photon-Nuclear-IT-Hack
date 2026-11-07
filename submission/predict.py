import argparse
import json
from pathlib import Path
from .backend import MAX_PNG, InputError, comparison, gray_bytes, predict

def main():
    parser= argparse.ArgumentParser(description =__doc__)
    parser.add_argument('--images', nargs =4, type=Path, required=True)
    parser.add_argument('--output', type=Path, required =True)
    parser.add_argument('--diagnostics-dir', type =Path)
    args =parser.parse_args()
    try:
        if args.output.suffix.lower()!= '.png':
            raise InputError('Выходной файл должен иметь расширение .png.')
        if any(p.resolve() ==args.output.resolve() for p in args.images):
            raise InputError('Выходной файл должен отличаться от входных.' )
        raw = []
        for path in args.images:
            with path.open('rb') as f:
                raw.append(f.read(MAX_PNG + 1))
        result= predict(raw, diagnostics=args.diagnostics_dir is not None)
        mask = result[ 0] if args.diagnostics_dir is not None else result
        args.output.parent.mkdir(parents=True, exist_ok =True)
        args.output.write_bytes( mask)
        if args.diagnostics_dir is not None:
            folder =args.diagnostics_dir
            folder.mkdir(parents=True, exist_ok=True)
            _, t, a , metrics= result
            for j in range( 4):
                (folder / f'reconstruction_{j+1}.png').write_bytes(gray_bytes(a[j]))
            (folder / 'comparison.png').write_bytes(comparison(t, a))
            (folder/ 'metrics.json').write_text(json.dumps(metrics , indent= 2), encoding= 'utf-8')
    except (InputError, OSError) as exc:
        parser.error(str(exc))
if __name__== '__main__':
    main()












    