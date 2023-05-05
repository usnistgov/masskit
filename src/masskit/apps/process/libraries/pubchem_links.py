#!/usr/bin/env python3
import sys
import json
import time
import math
from pathlib import Path
import requests
import hydra
from omegaconf import DictConfig, OmegaConf


def print_progress(curval=0, maxval=10, width=50):
    percent = int(curval/maxval*100)
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)

def use_cache(filename, source, *src_args):
    path = Path(filename).expanduser()
    if (not path.is_file()):
        Path(path.parent).mkdir(parents=True, exist_ok=True)
        data = source(*src_args)
        fresh_data = json.dumps(data)
        if len(fresh_data)>0:
            with path.open('w') as f:
                f.write(fresh_data)
    else:
        print(f"Using cache file {filename}")
    with path.open('r') as f:
        cache_data = json.load(f)
    return cache_data

# To get progress bar:
#   https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
def download_file(url, cfg):
    path = Path(cfg.cache.dir).expanduser()
    print(f"Downloading {url}\n    to {path}")
    path.mkdir(parents=True, exist_ok=True)
    local_file = path / url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with local_file.open('wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    print("Download complete")
    return local_file

class PubChemWiki:

    def __init__(self, cfg):
        self.cfg = cfg
        self.parse_json()

    def pubchem_wiki(self, session, page):
        parameters = {
            'source': 'Wikipedia',
            'heading_type': 'Compound',
            'heading': 'Wikipedia',
            'page': str(page),
            'response_type': 'save',
            'response_basename': 'PubChemAnnotations_Wikipedia'
        }
        r = session.get(self.cfg.wikipedia.urlbase, timeout=5, params=parameters)
        #print(r.url)
        #print(r.headers)
        r.raise_for_status()
        #print(f"  page: {rjson['Page']} of {rjson['TotalPages']}", end='\r')
        rjson = r.json()['Annotations']
        print_progress(rjson['Page'], rjson['TotalPages'])
        return rjson


    def get_pubchem_wiki(self):
        print("Fetching compounds with Wikipedia entries from PubChem:")
        annots = []
        with requests.Session() as s:
            rjson = self.pubchem_wiki(s, 1)
            annots.extend(rjson['Annotation'])
            total_pages = rjson['TotalPages']
            for pageno in range(2, total_pages+1):
                rjson = self.pubchem_wiki(s, pageno)
                annots.extend(rjson['Annotation'])
        print()
        return annots
    
    def parse_json(self):
        cache_file = f"{self.cfg.cache.dir}/{self.cfg.wikipedia.file}"
        wikidata = use_cache(cache_file, self.get_pubchem_wiki)
        #print(len(wikidata))

        self.cid2url = dict()
        for entry in wikidata:
            name = entry.get('Name')
            recs = entry.get('LinkedRecords')
            cid=None
            url = entry.get('URL')
            if recs: 
                cid_list = recs.get('CID')
                for cid in cid_list:
                    if cid in self.cid2url:
                        self.cid2url[cid].add(url)
                    else:
                        self.cid2url[cid] = set({url})
            #print(f"{name}\t{cid}\t{url}")
            #print(f"{url}")
        CIDs = set(self.cid2url.keys())

def get_pubchem_inchi(CIDs):
    print(f"Fetching InChI from PubChem for {len(CIDs)} CIDs:")
    annots = []
    num_groups = math.ceil(len(CIDs)/CID_GROUP_SIZE)
    cid_str = [str(x) for x in CIDs]

    with requests.Session() as s:
        for i in range(num_groups):
            cid = ",".join(cid_str[i:i+CID_GROUP_SIZE])
            req = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChI,InChIKey/JSON"
            r = s.get(req,timeout=5)
            r.raise_for_status()
            rjson = r.json()['PropertyTable']
            annots.extend(rjson['Properties'])
            print_progress(i,num_groups)
            time.sleep(1/CID_REQUESTS_PER_SECOND)
    print()
    return annots

# Join arrow tables:
# https://stackoverflow.com/questions/72122461/join-two-pyarrow-tables



@hydra.main(config_path="conf", config_name="config_pubchem_links", version_base=None)
def main(cfg: DictConfig) -> int:
    wikidata = PubChemWiki(cfg)
    print(len(wikidata.cid2url))
    #inchi_data = use_cache(CID_INCHI_FILE, get_pubchem_inchi, sorted(CIDs))
    download_file(cfg.pubchem.inchi.url, cfg)

    return 0

if __name__ == '__main__':
    sys.exit(main())
