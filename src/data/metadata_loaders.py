# src/data/loaders.py
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger


def clean_and_split_line(line: str, possible_delimiters: List[str] = None) -> Optional[List[str]]:
    """
    Cleans the line content and intelligently identifies delimiters

    ## Args:

    - line: The original line string

    - possible_delimiters: A list of possible delimiters, defaulting to [',', '|', ' ', '\t']

    ## Returns:

    - A list of cleaned fields; returns None if no delimiter can be identified.

    """
    if not line or not line.strip():
        return None

    # 1. Filter various quotation marks: ""、''、“”、‘’
    quote_pattern = r'["“”\'\'‘’]'  # Matches all types of quotes
    cleaned = re.sub(quote_pattern, "", line.strip())

    # 2. Default delimiter
    if possible_delimiters is None:
        possible_delimiters = [",", "|", " ", "\t"]

    # 3. Attempt to identify delimiters
    delimiter = None
    for delim in possible_delimiters:
        if delim in cleaned:
            # Avoid misinterpreting spaces (if there are only spaces but no other delimiters)
            if delim == " " and not any(d in cleaned for d in [",", "|", "\t"]):
                # Multiple consecutive spaces are used as a separator
                delimiter = r"\s+"
                break
            elif delim != " ":
                delimiter = delim
                break

    # 4. If no delimiter is detected, treat the entire line as a single field.
    if delimiter is None:
        return [cleaned]

    # 5. Split by delimiter
    if delimiter == r"\s+":
        fields = re.split(r"\s+", cleaned)
    else:
        fields = cleaned.split(delimiter)

    # 6. Clean up leading and trailing whitespace in each field
    fields = [f.strip() for f in fields if f.strip()]

    return fields


class BaseMetadataLoader(ABC):
    """The abstract base class for all dataset loaders"""

    @abstractmethod
    def load(self, meta_file: Path) -> pd.DataFrame:
        "A standardized DataFrame must be returned, containing the sample_id and mos columns."
        pass

    def _ensure_extension(self, df: pd.DataFrame, ext: str) -> pd.DataFrame:
        """Make sure sample_id has a specified file extension."""
        df["sample_id"] = df["sample_id"].apply(lambda x: x if x.lower().endswith(ext) else x + ext)
        return df

    def _parse_with_cleaner(self, meta_file: Path, delimiter_hint: List[str] = None) -> pd.DataFrame:
        """
        Use `clean_and_split_line` to parse the file line by line.
        Suitable for metadata files with non-standard formats.
        """
        records = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                fields = clean_and_split_line(line, delimiter_hint)
                if not fields:
                    continue

                # 假设前两列是 sample_id 和 mos
                if len(fields) >= 2:
                    records.append({"sample_id": fields[0], "mos": fields[1]})
                else:
                    logger.warning(f"行 {line_num} 字段数不足: {fields}")

        if not records:
            raise ValueError(f"未从 {meta_file} 解析到有效数据")

        return pd.DataFrame(records)


class Tid2013Loader(BaseMetadataLoader):
    def load(self, meta_file: Path) -> pd.DataFrame:
        # 使用 sep=r'\s+' 处理空格分隔，header=None 因为没有表头
        # names 明确指定列顺序：第一列是 MOS，第二列是 ID
        try:
            df = pd.read_csv(meta_file, sep=r"\s+", header=None, names=["mos", "sample_id"])
            df = self._ensure_extension(df, ".bmp")
            return df[["sample_id", "mos"]]

        except Exception as e:
            # If pandas read fails, use a cleanup function as a fallback.
            logger.warning(f"TID2013 pandas read failed, attempting to parse line by line: {e}")
            df = self._parse_with_cleaner(meta_file, delimiter_hint=[r"\s+"])
            df = self._ensure_extension(df, ".bmp")
            return df[["sample_id", "mos"]]


class KonvidLoader(BaseMetadataLoader):
    def load(self, meta_file: Path) -> pd.DataFrame:
        # Konvid 专用逻辑
        try:
            df = pd.read_csv(meta_file, quotechar='"', skipinitialspace=True)
            df = df.rename(columns={"flickr_id": "sample_id", "mos": "mos"})
            df["sample_id"] = df["sample_id"].astype(str).str.replace(r'["\s]', "", regex=True)
            df = self._ensure_extension(df, ".mp4")
            return df[["sample_id", "mos"]]

        except Exception as e:
            logger.warning(f"Konvid-1k pandas read failed, attempting to parse line by line: {e}")
            # Konvid is a comma separator.
            df = self._parse_with_cleaner(meta_file, delimiter_hint=[","])
            df = self._ensure_extension(df, ".mp4")
            return df[["sample_id", "mos"]]


class T2VqaLoader(BaseMetadataLoader):
    def load(self, meta_file: Path) -> pd.DataFrame:
        # T2VQA 专用逻辑
        try:
            df = pd.read_csv(meta_file, sep="|", header=None, names=["sample_id", "description", "mos"])
            df["description"] = df["description"].str.strip()
            df = self._ensure_extension(df, ".mp4")
            return df[["sample_id", "mos"]]
            # TODO:
            # return df[['sample_id', 'description', 'mos']]

        except Exception as e:
            logger.warning(f"T2VQA pandas read failed, attempting to parse line by line: {e}")
            # T2VQA is a vertical line separator.
            df = self._parse_with_cleaner(meta_file, delimiter_hint=["|"])
            # 注意：使用清洗函数时没有 description 列
            df = self._ensure_extension(df, ".mp4")
            return df[["sample_id", "mos"]]
