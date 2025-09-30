# -*- coding: utf-8 -*-
# EDA Agent (CSV/NF-e) — núcleo
import os, re, io, csv, json, unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

try:
    import chardet
except Exception:
    chardet = None
try:
    import duckdb
except Exception:
    duckdb = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def strip_accents(s: str) -> str:
    if not isinstance(s, str): return s
    nfkd = unicodedata.normalize('NFKD', s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def slugify_col(s: str) -> str:
    s = strip_accents(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def human_bytes(num: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024: return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"

def detect_encoding(path: Path, sample_size: int = 200_000) -> str:
    try: raw = Path(path).open('rb').read(sample_size)
    except Exception: return "utf-8"
    if chardet is not None:
        try:
            res = chardet.detect(raw)
            return res.get("encoding") or "utf-8"
        except Exception: pass
    for enc in ("utf-8", "latin-1", "cp1252"):
        try: raw.decode(enc); return enc
        except Exception: continue
    return "utf-8"

def sniff_delimiter(path: Path, encoding: str) -> Optional[str]:
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            sample = f.read(100_000)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        return None

_numeric_comma_regex = re.compile(r"^\s*-?\d{1,3}(\.\d{3})*,\d+\s*$")
_numeric_dot_regex   = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")
def looks_like_brazilian_number(x: str) -> bool:
    return isinstance(x, str) and bool(_numeric_comma_regex.match(x))
def normalize_brazilian_number(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if _numeric_comma_regex.match(s):
            s = s.replace(".", "").replace(",", ".")
            try: return float(s)
            except Exception: return np.nan
    return x

def try_parse_date_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind == "M": return s
    if s.dtype == "O":
        sample = s.dropna().astype(str).head(200)
        like = sum(1 for v in sample if re.search(r"\d{4}-\d{2}-\d{2}", v) or re.search(r"\d{2}/\d{2}/\d{4}", v))
        if (like / max(1, len(sample))) < 0.25: return s
        for dayfirst in (True, False):
            try: return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
            except Exception: pass
    return s

def is_valid_cnpj(cnpj: str) -> bool:
    if not isinstance(cnpj, str): return False
    c = re.sub(r"\D", "", cnpj)
    if len(c) != 14 or c == c[0]*14: return False
    def dv(digs: str) -> int:
        w = [5,4,3,2,9,8,7,6,5,4,3,2] if len(digs)==12 else [6,5,4,3,2,9,8,7,6,5,4,3,2]
        s = sum(int(d)*w[i] for i,d in enumerate(digs))
        r = s % 11
        return 0 if r < 2 else 11 - r
    d1 = dv(c[:12]); d2 = dv(c[:12] + str(d1));  return c[-2:] == f"{d1}{d2}"

def is_valid_cpf(cpf: str) -> bool:
    if not isinstance(cpf, str): return False
    c = re.sub(r"\D", "", cpf)
    if len(c) != 11 or c == c[0]*11: return False
    def dv(digs: str) -> int:
        s = sum(int(d)*w for d,w in zip(digs, range(len(digs)+1,1,-1)))
        r = (s*10) % 11
        return 0 if r == 10 else r
    d1 = dv(c[:9]); d2 = dv(c[:9] + str(d1));  return c[-2:] == f"{d1}{d2}"

def autowidth_fig(): return plt.figure(figsize=(9, 4.5), dpi=140)

@dataclass
class AgentConfig:
    prefer_value_cols: Tuple[str,...] = ("valor_total","valor","valor_empenhado","valor_liquidado","valor_pago","valor_da_nota","vl_total","vlr","preco","amount")
    prefer_date_cols: Tuple[str,...]  = ("data_emissao","data","emissao","competencia","data_documento","data_pagamento","data_liquidacao","transaction_date","time")
    prefer_cnpj_cols: Tuple[str,...]  = ("cnpj_emitente","cnpj","cnpj_fornecedor","cnpj_cpf","cnpj_cpf_fornecedor")
    prefer_cpf_cols: Tuple[str,...]   = ("cpf","cpf_fornecedor","cpf_cnpj","cpf_cnpj_fornecedor")
    prefer_desc_cols: Tuple[str,...]  = ("descricao","descricao_item","historico","item","objeto","natureza")
    max_charts_per_type: int = 12
    rounding: int = 2
    known_thresholds_brl: Tuple[float,...] = (8000.0, 17600.0, 50000.0)

class CSVAgent:
    def __init__(self, config: AgentConfig|None = None):
        self.config = config or AgentConfig()
        self.df: Optional[pd.DataFrame] = None
        self.original_columns: Optional[List[str]] = None
        self.colmap: Dict[str,str] = {}
        self.meta: Dict[str, Any] = {}

    def load(self, path_or_buffer) -> "CSVAgent":
        if isinstance(path_or_buffer, (str, Path)):
            path = Path(path_or_buffer)
            enc  = detect_encoding(path)
            delim= sniff_delimiter(path, enc)
            try: df = pd.read_csv(path, sep=delim if delim else None, engine="python", encoding=enc, dtype=str)
            except Exception: df = pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
        else:
            try: path_or_buffer.seek(0)
            except Exception: pass
            df = pd.read_csv(path_or_buffer, sep=None, engine="python", dtype=str)

        self.original_columns = list(df.columns)
        new_cols, self.colmap = [], {}
        for c in df.columns:
            sc = slugify_col(str(c))
            if sc in self.colmap:
                k=2; sc2=f"{sc}_{k}"
                while sc2 in self.colmap: k+=1; sc2=f"{sc}_{k}"
                sc = sc2
            self.colmap[sc] = c; new_cols.append(sc)
        df.columns = new_cols

        for col in df.columns:
            if df[col].dtype == "O":
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                sample = df[col].dropna().astype(str).head(200)
                frac = sum(looks_like_brazilian_number(v) for v in sample) / max(1, len(sample))
                if frac >= 0.5:
                    df[col] = df[col].map(normalize_brazilian_number)
                else:
                    frac_dot = sum(bool(_numeric_dot_regex.match(v)) for v in sample) / max(1, len(sample))
                    if frac_dot >= 0.5: df[col] = pd.to_numeric(df[col], errors="ignore")
        for col in df.columns: df[col] = try_parse_date_series(df[col])
        df = df.convert_dtypes()

        self.df = df
        self.meta["num_rows"] = len(df)
        self.meta["num_cols"] = len(df.columns)
        self.meta["memory_usage"] = df.memory_usage(deep=True
