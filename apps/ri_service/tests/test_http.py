import unittest
import json
import requests
import os

class HttpTests(unittest.TestCase):
    hostname =  '127.0.0.1'
    port = 5000
  
    def test_single_smiles(self):
        response = requests.get(f'http://{self.hostname}:{self.port}/molecule/smiles/CCCC/ri/json')
        output = response.json()
        print(output)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(output, {"CCCC":409.49939480000006})

    def test_single_smiles_canonical(self):
        response = requests.get(f'http://{self.hostname}:{self.port}/molecule/smiles/COC(=O)c1cccc(c1)C(=O)NC/ri/json')
        output = response.json()
        print(output)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(output, {"COC(=O)c1cccc(c1)C(=O)NC":1796.2127592})

    def test_multi_smiles(self):
        response = requests.get(f'http://{self.hostname}:{self.port}/molecule/smiles/CCCC,c1ccccc1/ri/json')
        output = response.json()
        print(output)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(output, {"CCCC":409.49939480000006,"c1ccccc1":648.6507752})

    def test_parse_smiles(self):
        response = requests.get(f'http://{self.hostname}:{self.port}/molecule/smiles/CCCC5654/ri/json')
        output = response.json()
        print(output)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(output, {"error": "unable to compute ri for ['CCCC5654']"})

    def test_format(self):
        response = requests.get(f'http://{self.hostname}:{self.port}/molecule/smiles/CCCC/ri/txt')
        output = response.json()
        print(output)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(output, {"error":"unknown file format txt"})


if __name__ == "__main__":
    HttpTests.hostname = os.environ.get('TEST_HOSTNAME', HttpTests.hostname)
    HttpTests.port = os.environ.get('TEST_PORT', HttpTests.port)
    unittest.main()
