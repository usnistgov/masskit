import unittest
import json

from ri_service import app


class BasicTests(unittest.TestCase):
  def setUp(self):
    self.app = app.test_client()
  
  def test_single_smiles(self):
        response = self.app.get('/molecule/smiles/CCCC/ri/json', follow_redirects=True)
        output = json.loads(response.data)
        print(output)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(output, {"CCCC":409.49939480000006})

  def test_single_smiles_canonical(self):
        response = self.app.get('/molecule/smiles/COC(=O)c1cccc(c1)C(=O)NC/ri/json', follow_redirects=True)
        output = json.loads(response.data)
        print(output)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(output, {"COC(=O)c1cccc(c1)C(=O)NC":1796.2127592})

  def test_multi_smiles(self):
        response = self.app.get('/molecule/smiles/CCCC,c1ccccc1/ri/json', follow_redirects=True)
        output = json.loads(response.data)
        print(output)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(output, {"CCCC":409.49939480000006,"c1ccccc1":648.6507752})

  def test_parse_smiles(self):
        response = self.app.get('/molecule/smiles/CCCC5654/ri/json', follow_redirects=True)
        output = json.loads(response.data)
        print(output)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(output, {"error": "unable to compute ri for ['CCCC5654']"})

  def test_format(self):
        response = self.app.get('/molecule/smiles/CCCC/ri/txt', follow_redirects=True)
        output = json.loads(response.data)
        print(output)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(output, {"error":"unknown file format txt"})


if __name__ == "__main__":
        unittest.main()
