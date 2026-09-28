#!/usr/bin/env python3
"""
patch_pocket_compose_labels.py

Safely replaces the single hardcoded _lab dict in pocket_compose.py with a
per-drug lookup, so ceritinib can have its 1986/2022 arrow labels positioned
independently of zidesamtinib/lorlatinib (whose figures are already correct).

Run once, from the repo root:
    python3 patch_pocket_compose_labels.py
It edits analysis/scripts/pocket_compose.py in place (after backing it up).
"""
import os, shutil, sys

TARGET = os.path.join('analysis', 'scripts', 'pocket_compose.py')

OLD = """    _lab = {'2022': dict(txt=(0.82, 0.72), tip=(0.74, 0.60)),
            '1986': dict(txt=(0.86, 0.26), tip=(0.80, 0.40))}"""

# Per-drug label positions (axes fractions). Defaults keep the tuned
# zidesamtinib/lorlatinib/cabozantinib layout (2022 high, 1986 low).
# ceritinib is FLIPPED (1986 high, 2022 low) and can be nudged independently:
#   txt = where the text sits, tip = where the arrowhead points (the cluster).
NEW = """    _LAB_DEFAULT = {'2022': dict(txt=(0.82, 0.72), tip=(0.74, 0.60)),
                    '1986': dict(txt=(0.86, 0.26), tip=(0.80, 0.40))}
    _LAB_BY_DRUG = {
        # ceritinib frame differs -> 1986 up, 2022 down; tune tip= to hit clusters
        'ceritinib': {'1986': dict(txt=(0.82, 0.72), tip=(0.74, 0.60)),
                      '2022': dict(txt=(0.86, 0.26), tip=(0.80, 0.40))},
    }
    _lab = _LAB_BY_DRUG.get(DRUG, _LAB_DEFAULT)"""


def main():
    if not os.path.exists(TARGET):
        sys.exit('Not found: %s  (run from the repo root)' % TARGET)
    src = open(TARGET).read()
    if 'ceritinib' in src and '_LAB_BY_DRUG' in src:
        sys.exit('Already patched (found _LAB_BY_DRUG). Nothing to do.')
    if OLD not in src:
        sys.exit('Could not find the exact _lab block to replace. '
                 'Paste lines 45-46 so the patch string can be matched.')
    shutil.copy(TARGET, TARGET + '.bak')
    open(TARGET, 'w').write(src.replace(OLD, NEW, 1))
    print('Patched %s  (backup at %s.bak)' % (TARGET, TARGET))
    print('ceritinib now uses independent label positions; edit _LAB_BY_DRUG to tune.')


if __name__ == '__main__':
    main()