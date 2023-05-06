#!/usr/bin/env python3
import sys
import json
import time
import math
from pathlib import Path
import requests
import hydra
from omegaconf import DictConfig, OmegaConf

from concurrent.futures import ThreadPoolExecutor
import signal
from functools import partial
from threading import Event
from typing import Iterable

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    track,
    TransferSpeedColumn,
)

class Download:

    def __init__(self, urls: Iterable[str], dest_dir: str):
        """Download multiple files to the given directory."""

        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            transient=True,
        )

        with self.progress:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for url in urls:
                    filename = url.split("/")[-1]
                    dest_path = dest_dir / filename
                    task_id = self.progress.add_task("download", filename=filename, start=False)
                    pool.submit(self.download_url, task_id, url, dest_path)


    def download_url(self, task_id: TaskID, url: str, path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            # The download display will break if the response doesn't contain content length
            total_size_in_bytes= int(r.headers.get('content-length', 0))
            self.progress.update(task_id, total=total_size_in_bytes)
            with path.open('wb') as f:
                self.progress.start_task(task_id)
                for chunk in r.iter_content(chunk_size=32768): 
                    f.write(chunk)
                    self.progress.update(task_id, advance=len(chunk))
        self.progress.console.log(f"Downloaded {path}")




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
        #print_progress(rjson['Page'], rjson['TotalPages'])
        return rjson


    def get_pubchem_wiki(self):
        annots = []
        with requests.Session() as s:
            rjson = self.pubchem_wiki(s, 1)
            annots.extend(rjson['Annotation'])
            total_pages = rjson['TotalPages']
            for pageno in track(range(2, total_pages+1), transient=True, description="Fetching Wikipedia entries:"):
                rjson = self.pubchem_wiki(s, pageno)
                annots.extend(rjson['Annotation'])
        return annots

    def use_cache(self, filename):
        path = Path(filename).expanduser()
        if (not path.is_file()):
            Path(path.parent).mkdir(parents=True, exist_ok=True)
            data = self.get_pubchem_wiki()
            fresh_data = json.dumps(data)
            if len(fresh_data)>0:
                with path.open('w') as f:
                    f.write(fresh_data)
        else:
            print(f"Using cache file {filename}")
        with path.open('r') as f:
            cache_data = json.load(f)
        return cache_data

    def parse_json(self):
        self.path = Path(self.cfg.cache.dir).expanduser()
        #cache_file = f"{self.cfg.cache.dir}/{self.cfg.wikipedia.file}"
        cache_file = self.path / self.cfg.wikipedia.file
        wikidata = self.use_cache(cache_file)

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
    #print(len(wikidata.cid2url))
    #inchi_data = use_cache(CID_INCHI_FILE, get_pubchem_inchi, sorted(CIDs))

    dlpath = Path(cfg.cache.dir).expanduser()
    dlpath.mkdir(parents=True, exist_ok=True)
    dlurls = []
    for key in cfg.pubchem.keys():
        dlurl = cfg.pubchem[key].url
        filename = dlurl.split('/')[-1]
        local_file = dlpath / filename
        if not local_file.is_file():
            dlurls.append(dlurl)
        else:
            print(f"Using cache file {local_file}")
    if len(dlurls) > 0:
        Download(dlurls, dlpath)

    return 0

if __name__ == '__main__':
    sys.exit(main())
