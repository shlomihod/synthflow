# https://github.com/sdv-dev/SDGym/blob/master/sdgym/synthesizers/privbn.py

import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def safe_f2i(fnum):
    inum = int(fnum)
    return inum if inum == fnum else fnum


class PrivBayes:
    """docstring for PrivBN."""

    def __init__(
        self, epsilon, epsilon_split=None, degree=None, theta=None, max_samples=None
    ):

        assert bool(degree is None) ^ bool(
            theta is None
        ), "Only one of `degree` or `theta` should be given."  # logical xor
        if degree is not None:
            self.mode = "degree"
            self.parameter = degree
        else:
            self.mode = "theta"
            self.parameter = theta

        self.epsilon = safe_f2i(epsilon)
        self.epsilon_split = epsilon_split
        privbayes_path_bin = os.getenv(
            "PRIVBAYES_BIN", "privbayes"
        )
        self.privbayes_bin = os.path.join(privbayes_path_bin, f"privBayes{self.mode.title()}.bin")
        if not os.path.exists(self.privbayes_bin):
            raise RuntimeError("privbayes binary not found. Please set PRIVBAYES_BIN")

        self.max_samples = max_samples

        self.transcript = None

    def fit_sample(self, data, ranges, categoricals, n_records=None):
        if n_records is None:
            n_records = len(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            os.makedirs(tmpdir / "data", exist_ok=True)
            os.makedirs(tmpdir / "log", exist_ok=True)
            os.makedirs(tmpdir / "output", exist_ok=True)
            LOGGER.info(f"Using {self.privbayes_bin}")
            shutil.copy(self.privbayes_bin, tmpdir / "privBayes.bin")
            d_cols = []
            with open(tmpdir / "data/real.domain", "w") as f:
                for id_, column in enumerate(data.columns):
                    if column in categoricals:
                        LOGGER.info(f"Column {column} is categorical")
                        print("D", end="", file=f)
                        counter = 0
                        size = ranges[column][1] - ranges[column][0] + 1
                        for i in range(size):
                            if i > 0 and i % 4 == 0:
                                counter += 1
                                print(" {", end="", file=f)
                            print("", i, end="", file=f)
                        print(" }" * counter, file=f)
                        d_cols.append(id_)
                    else:
                        LOGGER.info(f"Column {column} is not categorical")
                        minn = ranges[column][0]
                        maxx = ranges[column][1]
                        d = (maxx - minn) * 0.03
                        minn = minn - d
                        maxx = maxx + d
                        print("C", minn, maxx, file=f)

            with open(tmpdir / "data/real.dat", "w") as f:
                n = len(data)
                if self.max_samples:
                    n = min(n, self.max_samples)
                    data = data.sample(n=n, replace=False)
                else:
                    # shuffle data
                    data = data.sample(frac=1, replace=False)
                for i in range(n):
                    row = data.iloc[i]
                    for id_, col in enumerate(row):
                        if id_ in d_cols:
                            print(int(col), end=" ", file=f)

                        else:
                            print(col, end=" ", file=f)

                    print(file=f)

            privbayes = os.path.realpath(tmpdir / "privBayes.bin")
            arguments = [
                privbayes,
                "real",
                str(n_records),
                "1",
                str(self.epsilon),
                str(self.epsilon_split),
                str(self.parameter),
            ]
            LOGGER.info("Data Domain:")
            LOGGER.info((tmpdir / "data" / "real.domain").read_text())
            
            LOGGER.info("Calling %s", " ".join(arguments))
            start = datetime.utcnow()
            subprocess.call(arguments, cwd=tmpdir, stderr=subprocess.STDOUT)
            
            logs = (tmpdir / "log" / "real.log").read_text()
            LOGGER.info(logs)
            self.transcript = [logs]
            
            LOGGER.info("Elapsed %s", datetime.utcnow() - start)
            samples = np.loadtxt(
                tmpdir
                / (
                    f"output/syn_real_eps{int(self.epsilon)}"
                    f"_split{int(self.epsilon_split * 100)}"
                    f"_{self.mode}{self.parameter}_iter0.dat"
                )
            )
            return pd.DataFrame(samples, columns=data.columns)
