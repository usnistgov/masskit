#!/usr/bin/env python
import sys
import json
import time
import math

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable

import requests

import hydra
from omegaconf import DictConfig

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
from masskit.utils.general import MassKitSearchPathPlugin
from hydra.core.plugins import Plugins


Plugins.instance().register(MassKitSearchPathPlugin)


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
                    task_id = self.progress.add_task(
                        "download", filename=filename, start=False)
                    pool.submit(self.download_url, task_id, url, dest_path)

    def download_url(self, task_id: TaskID, url: str, path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            # The download display will break if the response doesn't contain content length
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            self.progress.update(task_id, total=total_size_in_bytes)
            with path.open('wb') as f:
                self.progress.start_task(task_id)
                for chunk in r.iter_content(chunk_size=32768):
                    f.write(chunk)
                    self.progress.update(task_id, advance=len(chunk))
        self.progress.console.print(f"Downloaded {path}")


class PubChemCAS:

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
        self.cas_schema = pa.schema([
            pa.field("cid", pa.int64()),
            pa.field("cas", pa.string()),
            pa.field("name", pa.string())
        ])
        self.parse_pubchem_json()

    def pubchem_cas(self, session, page):
        parameters = {
            'source': 'CAS Common Chemistry',
            'heading_type': 'Compound',
            'heading': 'CAS',
            'page': str(page),
            'response_type': 'save',
            'response_basename': 'PubChemAnnotations_CAS'
        }
        r = session.get(self.cfg.queries.cas.urlbase, timeout=5, params=parameters)
        #print(r.url)
        #print(r.headers)
        r.raise_for_status()
        rjson = r.json()['Annotations']
        return rjson

    def get_pubchem_cas(self):
        annots = []
        with self.progress:
            task_id = self.progress.add_task(
                "download",
                filename=self.cfg.queries.cas.file, 
                start=False,
            )
            with requests.Session() as s:
                rjson = self.pubchem_cas(s, 1)
                annots.extend(rjson['Annotation'])
                total_pages = rjson['TotalPages']
                self.progress.update(task_id, total=total_pages)
                self.progress.start_task(task_id)
                self.progress.update(task_id, advance=1)
                for pageno in range(2, total_pages+1):
                    rjson = self.pubchem_cas(s, pageno)
                    annots.extend(rjson['Annotation'])
                    self.progress.update(task_id, advance=1)
            self.progress.console.print(f"Downloaded {self.cfg.queries.cas.file}")
        return annots

    def use_pubchem_cache(self, filename):
        path = Path(filename).expanduser()
        if (not path.is_file()):
            Path(path.parent).mkdir(parents=True, exist_ok=True)
            data = self.get_pubchem_cas()
            fresh_data = json.dumps(data)
            if len(fresh_data) > 0:
                with path.open('w') as f:
                    f.write(fresh_data)
        else:
            self.progress.console.print(f"Using cache file {filename}")
        with path.open('r') as f:
            cache_data = json.load(f)
        return cache_data

    def parse_pubchem_json(self):
        path = Path(self.cfg.cache.dir).expanduser()

        parquet_file = path / self.cfg.queries.cas.parquet
        if parquet_file.is_file():
            self.progress.console.print(f"Using cache file {parquet_file}")
            return

        cache_file = path / self.cfg.queries.cas.file
        casdata = self.use_pubchem_cache(cache_file)

        records = self.cas_schema.empty_table().to_pydict()
        for entry in casdata:
            cas = entry.get('SourceID')
            name = entry.get('Name')
            recs = entry.get('LinkedRecords')
            if recs:
                cid_list = recs.get('CID')
                for cid in cid_list:
                    records["cid"].append(cid)
                    records["cas"].append(cas)
                    records["name"].append(name)
        table = pa.table(records, self.cas_schema)
        pq.write_table(table, parquet_file)


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
        self.parse_pubchem_json()
        self.counts_schema = pa.schema([
            pa.field("cid", pa.int64()),
            pa.field("pageviews", pa.int64()),
            pa.field("months", pa.int64()),
            pa.field("references", pa.int64()),
            pa.field("average_pageviews", pa.float32())
        ])
        self.fetch_counts()

    def pubchem_wiki(self, session, page):
        parameters = {
            'source': 'Wikipedia',
            'heading_type': 'Compound',
            'heading': 'Wikipedia',
            'page': str(page),
            'response_type': 'save',
            'response_basename': 'PubChemAnnotations_Wikipedia'
        }
        r = session.get(self.cfg.queries.wikipedia.pubchem_urlbase, timeout=5, params=parameters)
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
                filename=self.cfg.queries.wikipedia.file, 
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
            self.progress.console.print(f"Downloaded {self.cfg.queries.wikipedia.file}")
        return annots

    def use_pubchem_cache(self, filename):
        path = Path(filename).expanduser()
        if (not path.is_file()):
            Path(path.parent).mkdir(parents=True, exist_ok=True)
            data = self.get_pubchem_wiki()
            fresh_data = json.dumps(data)
            if len(fresh_data) > 0:
                with path.open('w') as f:
                    f.write(fresh_data)
        else:
            self.progress.console.print(f"Using cache file {filename}")
        with path.open('r') as f:
            cache_data = json.load(f)
        return cache_data

    def parse_pubchem_json(self):
        self.path = Path(self.cfg.cache.dir).expanduser()
        cache_file = self.path / self.cfg.queries.wikipedia.file
        wikidata = self.use_pubchem_cache(cache_file)

        self.cid2url = dict()
        for entry in wikidata:
            name = entry.get('Name')
            recs = entry.get('LinkedRecords')
            cid = None
            url = entry.get('URL')
            if recs:
                cid_list = recs.get('CID')
                for cid in cid_list:
                    if cid in self.cid2url:
                        self.cid2url[cid].add(url)
                    else:
                        self.cid2url[cid] = set({url})
        CIDs = set(self.cid2url.keys())

    def parse_counts(self, wikijson):
        items = wikijson['items']
        total = 0
        months = 0
        for item in items:
            total += int(item['views'])
            months += 1
        return (total, months)

    def fetch_counts(self):
        parquet_file = Path(self.cfg.cache.dir).expanduser() / \
            self.cfg.queries.wikipedia.parquet
        if parquet_file.is_file():
            self.progress.console.print(f"Using cache file {parquet_file}")
            return
        # Even though today might not be a full month, they only return results up to the previous full month.
        today = datetime.today().strftime('%Y%m%d')
        records = self.counts_schema.empty_table().to_pydict()
        with self.progress:
            task_id = self.progress.add_task(
                "download",
                filename="Wikipedia Page Views",
                start=False,
            )
            with requests.session() as session:
                session.headers.update(
                    {'User-Agent': self.cfg.queries.wikipedia.user_agent})
                self.progress.update(task_id, total=len(self.cid2url))
                self.progress.start_task(task_id)
                for k, v in self.cid2url.items():
                    total = 0
                    months = 0
                    references = 0
                    for url in v:
                        parts = urlparse(url)
                        # Can't use the following beacuse of pages like:
                        #    https://en.wikipedia.org/wiki/Tegafur/gimeracil/oteracil
                        # chemical = Path(parts.path).name
                        chemical = parts.path.replace(
                            "/wiki/", "", 1).replace("/", "%2F")
                        # Some links which look like the following don't work:
                        #    https://en.wikipedia.org/w/index.php?title=Diammonium_dioxido(dioxo)molybdenum&action=edit&redlink=1
                        if "index.php" in chemical:
                            continue
                        req_url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{parts.hostname}/all-access/user/{chemical}/monthly/20150701/{today}'
                        r = session.get(req_url, timeout=10)
                        r.raise_for_status()
                        counts = self.parse_counts(r.json())
                        total += counts[0]
                        months += counts[1]
                        references += 1
                        time.sleep(
                            1/self.cfg.queries.wikipedia.reqs_per_second)
                    self.progress.update(task_id, advance=1)
                    records["cid"].append(k)
                    records["pageviews"].append(total)
                    records["months"].append(months)
                    records["references"].append(references)
                    if months == 0:
                        records["average_pageviews"].append(0)
                    else:
                        records["average_pageviews"].append(total/months)

        table = pa.table(records, self.counts_schema)
        pq.write_table(table, parquet_file)

# Obsolete function to get the InChI data for a set of CIDs
# Now we are downloading the complete list from a file they publish.
# def get_pubchem_inchi(CIDs, cfg):
#     print(f"Fetching InChI from PubChem for {len(CIDs)} CIDs:")
#     annots = []
#     num_groups = math.ceil(len(CIDs)/cfg.queries.cid2inchi.group_size)
#     cid_str = [str(x) for x in CIDs]

#     with requests.Session() as s:
#         for i in track(range(num_groups), transient=False, description="Fetching PubChem InChI entries:"):
#             cid = ",".join(cid_str[i:i+cfg.queries.cid2inchi.group_size])
#             req = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChI,InChIKey/JSON"
#             r = s.get(req,timeout=5)
#             r.raise_for_status()
#             rjson = r.json()['PropertyTable']
#             annots.extend(rjson['Properties'])
#             time.sleep(1/cfg.queries.cid2inchi.reqs_per_second)
#     print()
#     return annots

class PubChemFTP:

    def __init__(self, cfg):
        self.cfg = cfg
        self.cache_pubchem_files(self.cfg.pubchem)

    def get_csv_type(self, t):
        if t == 'int8':
            return pa.int8()
        if t == 'int64':
            return pa.int64()
        
        return pa.string()

    def get_convert_options(self, col_types):
        col_types = {}
        for field in col_types:
            col_types[field['name']] = self.get_csv_type(field['type'])
        return pv.ConvertOptions(column_types=col_types)

    def cache_pubchem_files(self, cfg: DictConfig):
        dlpath = Path(self.cfg.cache.dir).expanduser()
        dlpath.mkdir(parents=True, exist_ok=True)
        dlurls = []
        csv_files = {}

        # Check which files need downloaded
        for key in cfg.keys():
            dlurl = cfg[key].url
            filename = dlurl.split('/')[-1]
            csv_file = dlpath / filename
            if not csv_file.is_file():
                dlurls.append(dlurl)
            else:
                global_console.print(f"Using cached source file: {csv_file}")
            csv_files[key] = csv_file

        # Perform required downloads
        if len(dlurls) > 0:
            Download(dlurls, dlpath)

        # Transform CSV file to Parquet
        for key in cfg.keys():
            pqfile = dlpath / cfg[key].parquet
            if not pqfile.is_file():
                csv_file = csv_files[key]
                global_console.print(f"transforming {csv_file.name} -> {pqfile.name}")
                convert_opts = self.get_convert_options(cfg[key].types)
                read_opts = pv.ReadOptions(column_names=cfg[key].headers)
                old_column_names = read_opts.column_names
                new_column_names = self.get_new_columns(old_column_names, cfg[key])
                print(f"\t{old_column_names} -> {new_column_names}")
                table = pv.read_csv(csv_file, 
                                    parse_options=pv.ParseOptions(delimiter='\t'),
                                    convert_options=convert_opts,
                                    read_options=read_opts
                                    )
                #global_console.print(table)
                table = self.process(table, cfg[key], new_column_names)
                pq.write_table(table, pqfile)
            else:
                global_console.print(f"Using cached processed file: {pqfile}")

    def get_new_columns(self, old_columns, cfg: DictConfig):
        if "process" in cfg:
            new_columns = list(cfg.process.groupby)
            new_columns.append(cfg.process.agg_newname)
            return new_columns
        else:
            return old_columns

    def process(self, table: pa.Table, cfg: DictConfig, column_names):
        if "process" in cfg:
            agg_list = [(cfg.process.aggregate, cfg.process.type)]
            sort_order = []
            for name in cfg.process.groupby:
                sort_order.append( (name, "ascending") )
            table = table.group_by(cfg.process.groupby).aggregate(agg_list).sort_by(sort_order).rename_columns(column_names)
        return table

class Analyze:

    def __init__(self, cfg):
        self.cfg = cfg
        self.cache_path = Path(self.cfg.cache.dir).expanduser()
        self.load_nist_data()
        self.load_cas()
        self.wikipedia_counts()
        self.pmid_counts()
        self.patent_counts()
        self.do_joins()
        self.save_data()

    def load_nist_data(self):
        tables = []
        for file in self.cfg.nist.files:
            table = pq.read_table(file, columns=['id', 'name', 'inchi_key'])
            tables.append(table)
        tables
        all_inchi_keys = pa.concat_arrays(
            [i.column("inchi_key").combine_chunks() for i in tables])
        inchi_keys = set(all_inchi_keys.unique().to_pylist())

        # Find the overlapping set with PubChem
        cid2inchi_file = self.cache_path / self.cfg.pubchem.inchi.parquet
        cid2inchi_full = pq.read_table(
            cid2inchi_file, columns=['cid', 'inchi_key'])
        table2 = pa.table({'inchi_key': inchi_keys})
        self.cid2inchi = cid2inchi_full.join(
            table2, keys='inchi_key', join_type='inner')
        print(
            f"InChI keys matched: {self.cid2inchi.num_rows} out of {len(inchi_keys)}.")

    def load_cas(self):
        cid2cas_file = self.cache_path / self.cfg.queries.cas.parquet
        self.cid2cas = pq.read_table(cid2cas_file)
        table = self.cid2inchi.join(
            self.cid2cas, keys='cid', join_type='inner')
        print(f"cas counts, num rows: {table.num_rows}")

    def pmid_counts(self):
        cid2pubmed_file = self.cache_path / self.cfg.pubchem.pmid.parquet
        cid2pmid_full = pq.read_table(cid2pubmed_file, columns=['cid', 'pmid'])
        cid2pmid_matched = self.cid2inchi.join(
            cid2pmid_full, keys='cid', join_type='inner')
        self.cid2pmid = cid2pmid_matched.group_by(
            ['cid', 'inchi_key']).aggregate([("pmid", "count_distinct")])
        print(f"pmid counts, num rows: {self.cid2pmid.num_rows}")

    def patent_counts(self):
        cid2patent_file = self.cache_path / self.cfg.pubchem.patent.parquet
        cid2patent_full = pq.read_table(cid2patent_file)

        # The Patent table is too big, so we need to join by parts
        sz = 50000000
        tables = []
        for i in track(range(0, cid2patent_full.num_rows, sz)):
            subtbl = cid2patent_full.slice(offset=i, length=sz)
            jointbl = self.cid2inchi.join(
                subtbl, keys='cid', join_type='inner')
            tables.append(jointbl)
            # print(f"Working on row numbers {i:,d} through {i+sz-1:,d}")
        cid2patent_matched = pa.concat_tables(tables)
        self.cid2patent = cid2patent_matched.group_by(
            ['cid', 'inchi_key']).aggregate([("patent_id", "count")])
        print(f"patent counts, num rows: {self.cid2patent.num_rows}")

    def wikipedia_counts(self):
        cid2wikipedia_file = self.cache_path / self.cfg.queries.wikipedia.parquet
        self.cid2wikipedia = pq.read_table(cid2wikipedia_file, columns=[
                                           'cid', 'average_pageviews'])
        print(f"Wikipedia counts, num rows: {self.cid2wikipedia.num_rows}")
        # wiki_counts = cid2wikipedia.sort_by([("average_pageviews","descending")])

    def do_joins(self):
        table = self.cid2inchi.join(
            self.cid2cas, keys='cid', join_type='left outer')
        table = table.join(self.cid2wikipedia, keys='cid',
                           join_type='left outer')
        table = table.join(self.cid2pmid, keys=[
                           'cid', 'inchi_key'], join_type='left outer')
        table = table.join(self.cid2patent, keys=[
                           'cid', 'inchi_key'], join_type='left outer')

        self.data = table.sort_by([("cid", "ascending")])

    def save_data(self):
        pq.write_table(self.data, "pubchem_links.parquet")

# Join arrow tables:
# https://stackoverflow.com/questions/72122461/join-two-pyarrow-tables


@hydra.main(config_path="conf", config_name="config_pubchem_links", version_base=None)
def main(cfg: DictConfig) -> int:
    # for k in cfg.pubchem.keys():
    #     print(k)
    #     print(cfg.pubchem[k].types)
    #     if "process" in cfg.pubchem[k]:
    #         print(cfg.pubchem[k].process)    
    # return 0
    global_console.print("Attempting to find or download data files:")
    casdata = PubChemCAS(cfg)
    wikidata = PubChemWiki(cfg)
    pubchemdata = PubChemFTP(cfg)

    #res = Analyze(cfg)

    return 0


if __name__ == '__main__':
    sys.exit(pubchem_links_app())
