import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from proxy_tasks.final16.catalog import build_part


class IncomingDataTests(unittest.TestCase):
    def test_emnist_idx_orientation_and_label_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            part = build_part('T2-R3-02')
            part.data_dir_override = Path(tmp)
            images = np.arange(2*28*28, dtype=np.uint16).astype(np.uint8).reshape(2,28,28)
            with gzip.open(Path(tmp)/'emnist-letters-train-images-idx3-ubyte.gz','wb') as stream:
                stream.write(struct.pack('>IIII',2051,2,28,28)+images.tobytes())
            with gzip.open(Path(tmp)/'emnist-letters-train-labels-idx1-ubyte.gz','wb') as stream:
                stream.write(struct.pack('>II',2049,2)+bytes([1,26]))
            part.prepare()
            x,y=part._read()
            np.testing.assert_array_equal(y,[0,25])
            self.assertEqual(x[0,3,7],images[0,7,3])
            self.assertEqual(x[1,19,5],images[1,5,19])
            self.assertEqual(part.out_dim,26)

    def test_lipophilicity_preparation_uses_scaffolds_and_preserves_csv(self):
        from validation.prepare import prepare_task
        from validation.runner import load_protocol
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source/round2/t2m_lipo'
            source.mkdir(parents=True)
            content = 'smiles,exp\nCC,1.0\nCCC,2.0\nc1ccccc1,3.0\nC1CCCCC1,4.0\n'
            (source / 'Lipophilicity.csv').write_text(content)
            target = root / 'data/T2-R2-02'
            prepare_task(root / 'source', target, 'T2-R2-02', load_protocol('T2-R2-02')[1])
            self.assertEqual((source / 'Lipophilicity.csv').read_text(), content)
            split = np.load(target / 'split.npz')
            ids = np.concatenate([split[k] for k in ['train', 'val', 'test']])
            np.testing.assert_array_equal(np.sort(ids), np.arange(4))
            self.assertTrue(any(0 in split[k] and 1 in split[k] for k in split.files))


if __name__ == '__main__':
    unittest.main()
