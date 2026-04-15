#?------------------------------------------------------------
#? Author: Nolan Tuttle & Mathew Hobson
#? Project: ResistorClassification
#?------------------------------------------------------------

import os
import cv2 as cv
import shutil as sh
from contour import isolate_band_region


def preprocess_dir(src, dst='archive_clean'):
    print(f'Copying {src} -> {dst} ...')
    sh.copytree(src, dst, dirs_exist_ok=True)

    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(dst)
        for f in files if f.endswith('.jpg')
    ]
    total   = len(all_files)
    count   = 0
    skipped = 0

    print(f'Processing {total} images...\n')

    for filepath in all_files:
        count += 1

        image = cv.imread(filepath)
        if image is None:
            print(f'  [SKIP] unreadable: {filepath}')
            os.remove(filepath)
            skipped += 1
            continue

        try:
            band_crop, _ = isolate_band_region(filepath)
            band_crop    = cv.resize(band_crop, (700, 700))
            cv.imwrite(filepath, band_crop)
        except Exception as e:
            print(f'  [SKIP] {os.path.basename(filepath)} — {e}')
            os.remove(filepath)
            skipped += 1
            continue

        if count % 50 == 0 or count == total:
            pct = count / total * 100
            print(f'  {count}/{total}  ({pct:.1f}%)  skipped: {skipped}')

    print(f'\nDone. Saved: {count - skipped}, Skipped: {skipped}, Total: {total}')


if __name__ == '__main__':
    preprocess_dir('archive')
