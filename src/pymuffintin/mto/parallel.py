"""MPI task ownership and node-local arrays for one NMTO iteration.

The caller owns MPI initialization and the supplied communicator.  Native
calculations remain rank-local; only node leaders transfer shared buffers
between nodes.  Arrays allocated here must not outlive this context.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import numpy as np
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
    from mpi4py import MPI


class NmtoParallel:
    def __init__(self, comm: MPI.Comm) -> None:
        from mpi4py import MPI

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=self.rank)
        self.leaders = comm.Split(
            0 if self.node.Get_rank() == 0 else MPI.UNDEFINED, key=self.rank
        )
        leader = self.node.bcast(self.rank if self.node.Get_rank() == 0 else None)
        rank_leaders = comm.allgather(leader)
        leader_indices = {rank: index for index, rank in enumerate(sorted(set(rank_leaders)))}
        self._owner_nodes = [leader_indices[rank] for rank in rank_leaders]
        self._windows: dict[int, tuple[NDArray, MPI.Win]] = {}

    def __enter__(self) -> NmtoParallel:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        from mpi4py import MPI

        for _, window in reversed(tuple(self._windows.values())):
            window.Unlock_all()
            window.Free()
        self._windows.clear()
        if self.leaders != MPI.COMM_NULL:
            self.leaders.Free()
        self.node.Free()

    def indices(self, count: int) -> range:
        return range(self.rank, count, self.size)

    def point_slice(self, count: int) -> slice:
        return slice(count * self.rank // self.size, count * (self.rank + 1) // self.size)

    def shared_array(self, shape: tuple[int, ...], dtype: DTypeLike) -> NDArray:
        from mpi4py import MPI

        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize if self.node.Get_rank() == 0 else 0
        window = MPI.Win.Allocate_shared(nbytes, dtype.itemsize, comm=self.node)
        window.Lock_all()
        buffer, _ = window.Shared_query(0)
        array = np.ndarray(shape, dtype=dtype, buffer=buffer)
        self._windows[id(array)] = (array, window)
        return array

    def _sync(self, array: NDArray) -> None:
        _, window = self._windows[id(array)]
        window.Sync()
        self.node.Barrier()
        window.Sync()

    def publish(self, array: NDArray) -> None:
        """Publish leading-axis tasks written by global rank ``index % size``."""
        self._sync(array)
        if self.node.Get_rank() == 0:
            for index in range(len(array)):
                self.leaders.Bcast(array[index:index + 1], root=self._owner_nodes[index % self.size])
        self._sync(array)

    def publish_rows(self, array: NDArray) -> None:
        """Publish contiguous sample blocks written using :meth:`point_slice`."""
        self._sync(array)
        if self.node.Get_rank() == 0:
            for rank in range(self.size):
                start, stop = len(array) * rank // self.size, len(array) * (rank + 1) // self.size
                self.leaders.Bcast(array[start:stop], root=self._owner_nodes[rank])
        self._sync(array)

    @contextmanager
    def local_stage(self) -> Iterator[None]:
        """Propagate a rank-local computation failure before the next collective."""
        error = None
        try:
            yield
        except Exception as caught:
            error = caught
        messages = self.comm.allgather(
            None if error is None else f"rank {self.rank}: {type(error).__name__}: {error}"
        )
        failure = next((message for message in messages if message is not None), None)
        if failure is not None:
            raise RuntimeError(f"NMTO MPI stage failed on {failure}") from error
