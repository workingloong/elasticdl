import os
from abc import ABC, abstractmethod
from contextlib import closing

import recordio

from elasticdl.python.common.odps_io import ODPSReader


class AbstractDataReader(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def read_records(self, task):
        """This method will be used in `TaskDataService` to read the records
        based on the information provided for a given task into a Python
        generator/iterator.

        Arguments:
            task: The current `Task` object that provides information on where
                to read the data for this task.
        """
        pass

    @abstractmethod
    def create_shards(self):
        """This method creates the dictionary of shards where the keys
        are the shard names and the values are the number of records.
        """
        pass


class RecordIODataReader(AbstractDataReader):
    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs
        if "data_dir" not in self._kwargs:
            raise ValueError("data_dir is required for RecordIODataReader()")

    def read_records(self, task):
        with closing(
            recordio.Scanner(
                task.shard_name, task.start, task.end - task.start
            )
        ) as reader:
            while True:
                record = reader.record()
                if record:
                    yield record
                else:
                    break

    def create_shards(self):
        data_dir = self._kwargs["data_dir"]
        if not data_dir:
            return {}
        f_records = {}
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            with closing(recordio.Index(p)) as rio:
                f_records[p] = rio.num_records()
        return f_records


class ODPSDataReader(AbstractDataReader):
    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs
        self._reader = ODPSReader(**self._kwargs)

    def read_records(self, task):
        records = self._reader.read_batch(
            start=task.start,
            end=task.end,
            columns=None,
        )
        for record in records:
            yield record

    def create_shards(self):
        table_size = self._reader.get_table_size()
        records_per_task = self._kwargs["records_per_task"]
        shards = {}
        for shard_id in range(table_size / records_per_task):
            shards[shard_id] = records_per_task
        return shards
