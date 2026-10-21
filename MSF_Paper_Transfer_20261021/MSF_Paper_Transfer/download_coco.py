"""Official COCO2017 images/annotations, resumable, with owned-file verification."""
from pathlib import Path
import json
import os
import shutil
import time
import urllib.request
import zipfile

ROOT = Path('/root/autodl-tmp/datasets/coco2017_sota_transfer')
ROOT.mkdir(parents=True, exist_ok=True)
ARCHIVES = [('train2017.zip', 'https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip'),
            ('val2017.zip', 'https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip'),
            ('annotations_trainval2017.zip', 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip')]
def status(state, **extra):
    obj = {'state': state, 'time': time.time(), **extra}
    (ROOT / 'download_status.json').write_text(json.dumps(obj, indent=2))
    print(json.dumps(obj), flush=True)

for name, url in ARCHIVES:
    done = ROOT / (name + '.verified.json')
    if done.exists():
        continue
    archive, partial = ROOT / name, ROOT / (name + '.part')
    while not archive.exists():
        if shutil.disk_usage(str(ROOT)).free < 80 * 1024**3:
            status('waiting_disk_reserve_80GiB', archive=name)
            time.sleep(60)
            continue
        size = partial.stat().st_size if partial.exists() else 0
        try:
            request = urllib.request.Request(url, headers={'Range': 'bytes=%d-' % size} if size else {})
            status('connecting', archive=name, bytes=size, url=url)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = size > 0 and response.status == 206
                with partial.open('ab' if append else 'wb') as target:
                    progress = -1
                    while True:
                        block = response.read(8*1024*1024)
                        if not block:
                            break
                        target.write(block)
                        bucket = target.tell() // (256*1024*1024)
                        if bucket != progress:
                            progress = bucket
                            status('downloading', archive=name, bytes=target.tell())
                        if shutil.disk_usage(str(ROOT)).free < 60 * 1024**3:
                            raise RuntimeError('disk reserve crossed')
            with zipfile.ZipFile(str(partial)) as z:
                if z.testzip() is not None:
                    raise ValueError('ZIP CRC verification failed')
            os.replace(str(partial), str(archive))
        except Exception as exc:
            status('download_retry', archive=name, error=str(exc), bytes=partial.stat().st_size if partial.exists() else 0)
            time.sleep(60)
    with zipfile.ZipFile(str(archive)) as z:
        expanded = sum(i.file_size for i in z.infolist())
        if shutil.disk_usage(str(ROOT)).free < expanded + 60 * 1024**3:
            status('waiting_extract_space', archive=name)
            while shutil.disk_usage(str(ROOT)).free < expanded + 60 * 1024**3:
                time.sleep(60)
        for info in z.infolist():
            target = (ROOT / info.filename).resolve()
            if os.path.commonpath([str(ROOT.resolve()), str(target)]) != str(ROOT.resolve()):
                raise ValueError('unsafe archive member')
        z.extractall(str(ROOT))
    done.write_text(json.dumps({'url': url, 'archive_bytes': archive.stat().st_size, 'crc': 'passed'}, indent=2))
    status('archive_extracted', archive=name)

for split, expected in [('train', 118287), ('val', 5000)]:
    annotations = json.loads((ROOT / 'annotations' / ('instances_' + split + '2017.json')).read_text())
    assert len(annotations['images']) == expected
    missing = [i['file_name'] for i in annotations['images'] if not (ROOT / (split+'2017') / i['file_name']).is_file()]
    assert not missing, missing[:5]
(ROOT / 'ready.json').write_text(json.dumps({'train_images': 118287, 'val_images': 5000, 'annotation_image_membership': 'passed'}))
status('ready')
