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

    def test_snli_removes_unknown_labels_and_preserves_hypothesis_bytes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            pq.write_table(pa.table(dict(premise=['p0','p1','p2','drop'],
                hypothesis=['h0','h1','h2','drop'],label=[0,1,2,-1])),path/'train.parquet')
            premises=['앞'*150+'끝','p','p'];hypotheses=['h','비'*100,'h2']
            pq.write_table(pa.table(dict(premise=premises,hypothesis=hypotheses,label=[0,1,2])),path/'validation.parquet')
            part=build_part('T2-R2-01');part.data_dir_override=path;part.prepare()
            self.assertEqual(np.load(path/'train_x.npy').shape,(3,256))
            np.testing.assert_array_equal(np.load(path/'train_y.npy'),[0,1,2])
            x=np.load(path/'val_x.npy')
            expected=premises[0].encode('utf-8')[-254:]+b'\x1fh'
            self.assertEqual(bytes((x[0]-1).astype(np.uint8)),expected)
            expected=b'\x1f'+hypotheses[1].encode('utf-8')[:255]
            self.assertEqual(bytes((x[1]-1).astype(np.uint8)),expected)
            self.assertEqual(json.loads((path/'meta.json').read_text())['max_len'],256)


if __name__=='__main__':
    unittest.main()
