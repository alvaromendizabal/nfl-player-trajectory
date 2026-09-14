"""Regression tests use the real Round 1 float32 persistence contract."""
import io
import unittest
import numpy as np
from data_pipeline import verify_parent_storage_parity
from origin_features import ALL_NAMES,build_view,query_features
from test_pairs import fixture

class StorageParityTests(unittest.TestCase):
    def setUp(self):
        raw,keys=fixture()
        self.x,_=query_features(build_view(raw,origin=20),keys[:,2],keys[:,3]/10)
        self.saved=self.x.astype(np.float32)
    def test_reproduces_old_failure(self):
        self.assertFalse(np.allclose(self.x,self.saved,rtol=0,atol=1e-12))
    def test_exact_after_original_cast(self):
        self.assertTrue(verify_parent_storage_parity(self.x,self.saved)['exact_after_storage_conversion'])
    def test_actual_npz_roundtrip(self):
        stream=io.BytesIO();np.savez_compressed(stream,X=self.saved)
        with np.load(io.BytesIO(stream.getvalue()),allow_pickle=False) as z:
            verify_parent_storage_parity(self.x,z['X'])
    def test_one_float32_ulp_rejected(self):
        saved=self.saved.copy();saved[0,0]=np.nextafter(saved[0,0],np.float32(np.inf))
        with self.assertRaises(ValueError):verify_parent_storage_parity(self.x,saved)
    def test_different_row_order_rejected(self):
        with self.assertRaises(ValueError):verify_parent_storage_parity(self.x[::-1],self.saved)
    def test_different_feature_rejected(self):
        x=self.x.copy();x[0,2]+=.01
        with self.assertRaises(ValueError):verify_parent_storage_parity(x,self.saved)
    def test_float64_saved_rejected(self):
        with self.assertRaises(ValueError):verify_parent_storage_parity(self.x,self.saved.astype(np.float64))
    def test_float32_rebuilt_rejected(self):
        with self.assertRaises(ValueError):verify_parent_storage_parity(self.saved,self.saved)
    def test_nan_rejected(self):
        for target in ['x','saved']:
            x=self.x.copy();saved=self.saved.copy()
            if target=='x':x[0,0]=np.nan
            else:saved[0,0]=np.nan
            with self.assertRaises(ValueError):verify_parent_storage_parity(x,saved)
    def test_infinity_rejected(self):
        x=self.x.copy();x[0,0]=np.inf
        with self.assertRaises(ValueError):verify_parent_storage_parity(x,self.saved)
    def test_shape_rejected(self):
        with self.assertRaises(ValueError):verify_parent_storage_parity(self.x[:1],self.saved)
    def test_width_rejected(self):
        with self.assertRaises(ValueError):verify_parent_storage_parity(self.x[:,:-1],self.saved[:,:-1])
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):verify_parent_storage_parity(self.x[:0],self.saved[:0])
    def test_unchanged_inputs(self):
        a=self.x.tobytes();b=self.saved.tobytes()
        self.x.flags.writeable=False;self.saved.flags.writeable=False
        verify_parent_storage_parity(self.x,self.saved)
        self.assertEqual(self.x.tobytes(),a);self.assertEqual(self.saved.tobytes(),b)
    def test_quantization_not_arbitrary_tolerance(self):
        saved=np.ones((2,len(ALL_NAMES)),np.float32)
        x=saved.astype(np.float64)+1e-9
        verify_parent_storage_parity(x,saved)
        x[0,0]=float(np.nextafter(saved[0,0],np.float32(np.inf)))
        with self.assertRaises(ValueError):verify_parent_storage_parity(x,saved)

if __name__=='__main__':unittest.main()
