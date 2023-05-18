#!/usr/bin/env python3
import sys
import json
import time
import math

from collections import namedtuple
from pathlib import Path
from typing import Iterable

import requests

import hydra
from omegaconf import DictConfig, OmegaConf

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    track,
    TransferSpeedColumn,
)

global_console = Console()

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
            transient=False,
            console=global_console,
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
        self.progress.console.print(f"Downloaded {path}")


class PubChemWiki:

    def __init__(self, cfg):
        self.cfg = cfg
        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(),
            transient=False,
            console=global_console,
            )
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
        rjson = r.json()['Annotations']
        return rjson


    def get_pubchem_wiki(self):
        annots = []
        with self.progress:
            task_id = self.progress.add_task(
                "download",
                filename=self.cfg.wikipedia.file, 
                start=False,
                )
            with requests.Session() as s:
                rjson = self.pubchem_wiki(s, 1)
                annots.extend(rjson['Annotation'])
                total_pages = rjson['TotalPages']
                self.progress.update(task_id, total=total_pages)
                self.progress.start_task(task_id)
                self.progress.update(task_id, advance=1)
                for pageno in range(2, total_pages+1):
                    # description="Fetching Wikipedia entries:"):
                    rjson = self.pubchem_wiki(s, pageno)
                    annots.extend(rjson['Annotation'])
                    self.progress.update(task_id, advance=1)
            self.progress.console.print(f"Downloaded {self.cfg.wikipedia.file}")
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
            self.progress.console.print(f"Using cache file {filename}")
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

# Obsolete function to get the InChI data for a set of CIDs
# Now we are downloading the complete list from a file they publish.
def get_pubchem_inchi(CIDs, cfg):
    print(f"Fetching InChI from PubChem for {len(CIDs)} CIDs:")
    annots = []
    num_groups = math.ceil(len(CIDs)/cfg.queries.cid2inchi.group_size)
    cid_str = [str(x) for x in CIDs]

    with requests.Session() as s:
        for i in track(range(num_groups), transient=False, description="Fetching PubChem InChI entries:"):
            cid = ",".join(cid_str[i:i+cfg.queries.cid2inchi.group_size])
            req = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChI,InChIKey/JSON"
            r = s.get(req,timeout=5)
            r.raise_for_status()
            rjson = r.json()['PropertyTable']
            annots.extend(rjson['Properties'])
            time.sleep(1/cfg.queries.cid2inchi.group_size.reqs_per_second)
    print()
    return annots

def get_csv_type(t):
    if t == 'int8':
        return pa.int8()
    if t == 'int64':
        return pa.int64()
    
    return pa.string()

def get_convert_options(col_types):
    col_types = {}
    for field in col_types:
        col_types[field['name']] = get_csv_type(field['type'])
    return pv.ConvertOptions(column_types=col_types)

def cache_pubchem_files(cfg: DictConfig):
    dlpath = Path(cfg.cache.dir).expanduser()
    dlpath.mkdir(parents=True, exist_ok=True)
    dlurls = []
    transform_files = []

    for key in cfg.pubchem.keys():
        dlurl = cfg.pubchem[key].url
        filename = dlurl.split('/')[-1]
        csv_file = dlpath / filename
        if not csv_file.is_file():
            dlurls.append(dlurl)
        else:
            global_console.print(f"Using cache file {csv_file}")
        pqfile = dlpath / cfg.pubchem[key].parquet
        transform_files.append( (csv_file,pqfile,key) )        
    if len(dlurls) > 0:
        Download(dlurls, dlpath)
    for xfile in transform_files:
        if not xfile[1].is_file():
            global_console.print(f"transforming {xfile[0].name} -> {xfile[1].name}")
            convert_opts = get_convert_options(cfg.pubchem[xfile[2]].types)
            read_opts = pv.ReadOptions(column_names=cfg.pubchem[xfile[2]].headers)
            print(read_opts.column_names)
            table = pv.read_csv(xfile[0], 
                                parse_options=pv.ParseOptions(delimiter='\t'),
                                convert_options=convert_opts,
                                read_options=read_opts
                                )
            #global_console.print(table)
            pq.write_table(table, xfile[1])


# Join arrow tables:
# https://stackoverflow.com/questions/72122461/join-two-pyarrow-tables

@hydra.main(config_path="conf", config_name="config_pubchem_links", version_base=None)
def main(cfg: DictConfig) -> int:
    global_console.print("Attempting to find or download data files:")
    wikidata = PubChemWiki(cfg)
    cache_pubchem_files(cfg)

    return 0

if __name__ == '__main__':
    sys.exit(main())
