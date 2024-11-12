import re
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Literal

import pytz
import numpy as np
import pandas as pd


class DataLoader:
    _participants = [f"par_{i}" for i in range(1, 22)]
    _modalities = {
        "ema": ["daily", "weekly"],
        "oura": [
            "activity_level",
            "activity",
            "heart_rate",
            "readiness",
            "sleep_hypnogram",
            "sleep",
        ],
        "personicle": ["personicle"],
        "samsung": [
            "awake_time",
            "imu",
            "ppg",
            "pedometer",
            "pressure",
            "hrv_1min",
            "hrv_5min",
        ],
        "assessment": ["events", "surveys"],
    }

    def __init__(self, dataset_source: str | Path):
        """Constructor method

        :param dataset_source: The address of the zipfile for the dataset or 
            its unzipped directories root
        :type dataset_source: str | pathlib.Path
        :raises ValueError: When the source doesn't exist
        """
        self.source = Path(dataset_source)
        if not self.source.exists():
            raise ValueError("data_source file/directory not found.")
        if self.source.suffix not in (".zip", ""):
            raise ValueError("data_source must be a directory or a zip file.")
        self.is_zip = self.source.suffix == ".zip"

    def get_participants(self) -> list[str]:
        """get a list of all the participant ids

        :return: list of participant ids
        :rtype: list[str]
        """
        return deepcopy(self._participants)

    def get_modalities(self) -> dict[str, list[str]]:
        """get a dictionnary of all the modalities as keys and their files as values

        :return: modalities
        :rtype: dict[str, list[str]]
        """
        return deepcopy(self._modalities)

    @contextmanager
    def get_file(self, modality: str, file: str, participant: str):
        if participant not in self._participants:
            raise ValueError(f"Participant {participant} does not exist")
        if self.is_zip:
            try:
                source_zip = zipfile.ZipFile(self.source)
                ret = source_zip.open(f"{participant}/{modality}/{file}.csv")
                yield ret
            finally:
                ret.close()
                source_zip.close()
        else:
            try:
                ret = open(self.source / participant / modality / f"{file}.csv", "r")
                yield ret
            finally:
                ret.close()

    def file_exists(self, modality: str, file: str, participant: str) -> bool:
        if participant not in self._participants:
            return False
        if modality not in self._modalities or file not in self._modalities[modality]:
            return False
        if self.is_zip:
            with zipfile.ZipFile(self.source) as source_zip:
                return zipfile.Path(
                    source_zip, f"{participant}/{modality}/{file}.csv"
                ).exists()
        else:
            return (self.source / participant / modality / f"{file}.csv").exists()

    def load_single_df(
        self, modality: str, file: str, participant: str
    ) -> pd.DataFrame:
        with self.get_file(modality, file, participant) as f:
            return pd.read_csv(f)

    def load_df(
        self,
        modality: str,
        file: str,
        participants: str | list[str] = "all",
        add_id: bool = True,
        not_exist: Literal["ignore", "error"] = "ignore",
    ) -> pd.DataFrame:
        """load a DataFrame from the dataset

        :param modality: The modality to load
        :type modality: str
        :param file: The file to load from modality
        :type file: str
        :param participants: The name or list of the participants to load,
            defaults to 'all'
        :type participants: str, list[str], 'all'
        :param add_id: If you want to add the id of the participant as a column
            of the dataframe, defaults to True
        :type add_id: bool
        :param not_exist: What to do in case the file does not exist for a 
            participant, defaults to 'ignore'
        :type not_exist: 'ignore'|'error'
        :raises ValueError: When not_exist sets to 'error' and the file 
            could not be found.
        :return: The dataframe
        :rtype: pandas.DataFrame
        """
        if type(participants) is str:
            if participants == "all":
                participants = self._participants
            else:
                participants = [participants]
        df = pd.DataFrame()
        for par in participants:
            is_exist = self.file_exists(modality, file, par)
            if not is_exist and not_exist == "ignore":
                continue
            elif not is_exist and not_exist == "error":
                raise ValueError(
                    f"modailty {modality}, file {file} does not exist for participant {par}"
                )
            frag = self.load_single_df(modality, file, par)
            if add_id:
                frag["id"] = par
            df = pd.concat([df, frag], join="outer", ignore_index=True)
        return df.copy()


class DataTransform:
    @classmethod
    def ts_to_pydt(
        cls,
        timestamp: int,
        timezone: pytz.timezone = pytz.timezone("America/Los_Angeles"),
    ):
        return datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc).astimezone(
            timezone
        )

    @classmethod
    def datestr_to_pyd(cls, datestr: str):
        return date.fromisoformat(datestr)

    @classmethod
    def df_to_pydt(
        cls,
        df: pd.DataFrame,
        timezone: pytz.timezone = pytz.timezone("America/Los_Angeles"),
    ):
        """Convert dataframe 'date' columns to Python date object and dataframe 
            'timestamp' columns to Python datetime object.

        :param df: DataFrame to be processed inplace
        :type df: pandas.DataFrame
        :param timezone: timezone to convert timestamps to, 
            defaults to pytz.timezone('America/Los_Angeles')
        :type pytz.timezone

        :returns: None
        """
        cols = list(df.columns)
        ts_cols = filter(lambda col: "timestamp" in col, cols)
        for col in ts_cols:
            df[col.replace("timestamp", "pydt")] = df[col].apply(
                cls.ts_to_pydt, args=(timezone,)
            )
        cols = list(df.columns)
        ds_cols = filter(lambda col: "date" == col, cols)
        for col in ds_cols:
            df["pydate"] = df[col].apply(cls.datestr_to_pyd)
