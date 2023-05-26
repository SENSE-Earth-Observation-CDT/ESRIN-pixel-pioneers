#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import Filter
from typing import Callable, List, Any, Iterable, Optional

from tqdm.auto import tqdm

from sentinelhub import CRS

@dataclass
class BaseConfig:
    bucket_name: Optional[str]
    aws_region: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]

def multiprocess(process_fun: Callable, arguments: Iterable[Any],
                 total: Optional[int] = None, max_workers: int = 4) -> List[Any]:
    """
    Executes multiprocessing with tqdm.
    Parameters
    ----------
    process_fun: A function that processes a single item.
    arguments: Arguments with which te function is called.
    total: Number of iterations to run (for cases of iterators) 
    max_workers: Max workers for the process pool executor.

    Returns A list of results.
    -------
    

    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_fun, arguments), total=total))
    return results

def mgrs_to_utm(mgrs_tile_name: str) -> CRS:
    mgrs = mgrs_tile_name[:3]
    zone_id = int(mgrs[:2])
    north = mgrs[-1].upper() >= 'N'
    if north:
        base = 32600
    else:
        base = 32700
    return CRS(base + zone_id)
