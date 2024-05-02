from __future__ import annotations

import itertools as it
import logging
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from numbers import Number
from typing import Sequence

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from numpy.typing import ArrayLike  # type: ignore
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.sparse import lil_matrix  # type: ignore
from scipy.sparse.csgraph import maximum_bipartite_matching  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore
from tqdm import tqdm  # type: ignore

LOGGER = logging.getLogger(__name__)

ALGORITHMS = (
    "push_relabel_igraph",
    "push_relabel_graph_tool",
    "hopcroft_karp_scipy",
    "edmonds_karp_graph_tool",
)

HOPCROFT_KARP_SCIPY_MAX_DENSITY_THRESHOLD = 0.005  # .5%
HOPCROFT_KARP_SCIPY_MAX_SIZE_THRESHOLD = 50000

IGRAPH_AVAILEABLE = False
GRAPH_TOOL_AVAILEABLE = False
try:
    import igraph  # type: ignore

    IGRAPH_AVAILEABLE = True
except ModuleNotFoundError:
    LOGGER.warn("Could not import igraph. Trying to import graph-tool.")
    try:
        import graph_tool.all as gt  # type: ignore

        GRAPH_TOOL_AVAILEABLE = True
    except ModuleNotFoundError:
        LOGGER.warn(
            "Could not import graph-tool."
            " Some of the Faithfulness functionality won't work."
        )


def scale_columns_by_weights(
    x: pd.DataFrame,
    column_weights: dict[str, float],
) -> ArrayLike:
    """
    Scaling columns according to the given weights.

    Scaling is according to a number or based an index of in bins.

    Args:
        x (DataFrame): The DataFrame to be scaled.
        column_weights (dict): The scaling weight (number or bins) per column.

    Returns:
        DataFrame: The scaled DataFrame.

    Raises:
        TypeError: If the weight of a column is not valid.
    """

    x = x.copy()

    columns_with_weights = list(column_weights.keys())

    for column in columns_with_weights:
        weight = column_weights[column]

        if isinstance(weight, Number):
            x[column] *= weight

        elif not np.isscalar(weight):
            x[column] = np.digitize(x[column], weight, right=False)

        else:
            raise TypeError(
                f"Column weight `{column}` type is {type(weight)},"
                " but it shoulde be either number or a sequence of bins."
            )

    return x


def _compute_radius_neighbors(
    x: pd.DataFrame,
    y: pd.DataFrame,
    alpha: float,
    p: float,
    column_weights: None | dict[str, float] = None,
) -> tuple[ArrayLike, float]:
    """
    Find the neighbors within a radius of each row in the first DataFrame in the second.


    Args:
        x (DataFrame): The first DataFrame.
        y (DataFrame): The second DataFrame.
        alpha (float): The radious of the neighbors.
        p (alpha): Parameter for the Minkowski metric
                   from `sklearn.metrics.pairwise.pairwise_distances`.
        column_weights (dict, optional): The scaling weight (number or bins) per column.

    Returns:
        Array: The radius-neighbor indices of the second DataFrame
               for each row in first DataFrame.
        float: Density of the bipartite graph.
    """

    assert len(x) == len(y)

    if column_weights is not None:
        x = scale_columns_by_weights(x, column_weights)
        y = scale_columns_by_weights(y, column_weights)
    else:
        x, y = x.values, y.values

    neigh = NearestNeighbors(p=p)
    neigh.fit(x)
    _, inds = neigh.radius_neighbors(y, alpha)

    n_records = len(x)
    density = sum(map(len, inds)) / (n_records * (n_records - 1) / 2)

    return inds, density


def _build_graph(inds: Sequence[ArrayLike], directed: bool):
    n_records = len(inds)

    g = gt.Graph(directed=directed)

    for row, cols in enumerate(inds):
        edges = it.product([row], cols + n_records)
        if len(cols):
            g.add_edge_list(edges)

    assert gt.is_bipartite(g)

    return g


def _build_graph_mat(inds: Sequence[ArrayLike]) -> ArrayLike:
    n_records = len(inds)
    gmat = lil_matrix((n_records, n_records))

    for row, cols in enumerate(inds):
        if len(cols):
            gmat[row, sorted(cols)] = np.ones(len(cols), dtype=int)

    return gmat.tocsr()


def _extract_matching_info(g, pm, n_records):
    edge_and_ends = ((e, tuple(e)) for e in g.edges())

    matched_edges = [
        (int(tgt) - n_records, int(src))
        for e, (src, tgt) in edge_and_ends
        if int(src) < 2 * n_records and int(tgt) < 2 * n_records and pm[e] > 0
    ]

    matched_edges = sorted(matched_edges)

    matching_cardinality = len(matched_edges)

    first_matched, second_matched = zip(*matched_edges)
    matching = (first_matched, second_matched)

    vertex_indices = set(range(n_records))

    unmatched_indices = vertex_indices - set(first_matched), vertex_indices - set(
        second_matched
    )

    beta = 1 - matching_cardinality / n_records

    return beta, matching, unmatched_indices


def _compute_beta_push_relabel_graph_tool(
    inds: Sequence[ArrayLike],
) -> tuple[float, tuple[tuple], tuple[set]]:
    g = _build_graph(inds, directed=True)

    n_records = len(inds)

    source = 2 * n_records
    target = 2 * n_records + 1
    source_to_left = ((source, left) for left in range(n_records))
    right_to_target = ((right, target) for right in range(n_records, 2 * n_records))
    flow_edges = it.chain(source_to_left, right_to_target)
    g.add_edge_list(flow_edges)

    cap = g.new_edge_property("int")
    for e in g.edges():
        cap[e] = 1

    flow = gt.push_relabel_max_flow(g, g.vertex(source), g.vertex(target), cap)
    flow.a = cap.a - flow.a

    beta, matching, unmatched_indices = _extract_matching_info(g, flow, n_records)

    return beta, matching, unmatched_indices


def _compute_beta_edmonds_karp_graph_tool(inds: Sequence[ArrayLike]) -> float:
    n_records = len(inds)

    g = _build_graph(inds, directed=False)

    matching = gt.max_cardinality_matching(
        g, edges=True, bipartite=True, heuristic=False
    )

    beta, matching, unmatched_indices = _extract_matching_info(g, matching, n_records)

    return beta, matching, unmatched_indices


def _compute_beta_hopcroft_karp_scipy(
    inds: Sequence[ArrayLike],
) -> tuple[float, tuple[tuple], tuple[set]]:
    gmat = _build_graph_mat(inds)
    one_vector_matching = maximum_bipartite_matching(gmat)

    matched_mask = one_vector_matching != -1
    first_matched_indices = np.nonzero(matched_mask)[0]
    second_matched_indices = np.array(one_vector_matching)[first_matched_indices]
    matching = (tuple(first_matched_indices.astype(int)), tuple(second_matched_indices))

    unmatched_mask = one_vector_matching == -1
    beta = unmatched_mask.mean()
    unmatched_indices = (
        set(np.nonzero(unmatched_mask)[0]),
        set(range(len(inds))) - set(one_vector_matching[~unmatched_mask]),
    )

    return beta, matching, unmatched_indices


def _compute_beta_push_relabel_igraph(
    inds: Sequence[ArrayLike],
) -> tuple[float, tuple[tuple], tuple[set]]:
    n_records = len(inds)

    gmat = _build_graph_mat(inds)
    nz = gmat.nonzero()
    edges = zip(nz[0], nz[1] + n_records)

    n_records = len(inds)
    bipartite_mask = [False] * n_records + [True] * n_records
    g = igraph.Graph.Bipartite(bipartite_mask, edges, directed=False)

    matched_vertices = np.array(g.maximum_bipartite_matching().matching)[:n_records]

    second_matched_mask = matched_vertices[:n_records] != -1
    second_matched = np.nonzero(second_matched_mask)[0]
    first_matched = matched_vertices[second_matched] - n_records

    assert len(first_matched) == len(second_matched)
    matching_cardinality = len(first_matched)
    matching = (first_matched, second_matched)

    beta = 1 - matching_cardinality / n_records

    vertex_indices = set(range(n_records))
    unmatched_indices = vertex_indices - set(first_matched), vertex_indices - set(
        second_matched
    )

    return beta, matching, unmatched_indices


def _compute_beta_by_algorithm(
    inds: ArrayLike, algorithm: str
) -> tuple[float, tuple[tuple], tuple[set]]:
    """
    Compute the beta of fatihfulness given the adjency list between two tables.

    The beta is the proportion of not-matched pairs.

    The adjency list is based on the distance between every pair of rows
    and the alpha parameter.


    Args:
        Array (DataFrame): The adjency list.
                           I.e., The radius-neighbor indices of the second DataFrame
                           for each row in first DataFrame.
        algorithm (str): Which max cardinality bipartite matching algorithm to use.

    Returns:
        float: Beta value.
        float: The matching as two inideces sequences.
        tuple: The indices of the unmatched rows.
    """

    if algorithm == "push_relabel_igraph":
        assert IGRAPH_AVAILEABLE
        beta, matching, unmatched_indices = _compute_beta_push_relabel_igraph(inds)

    elif algorithm == "push_relabel_graph_tool":
        assert GRAPH_TOOL_AVAILEABLE
        beta, matching, unmatched_indices = _compute_beta_push_relabel_graph_tool(inds)

    elif algorithm == "edmonds_karp_graph_tool":
        assert GRAPH_TOOL_AVAILEABLE
        beta, matching, unmatched_indices = _compute_beta_edmonds_karp_graph_tool(inds)

    elif algorithm == "hopcroft_karp_scipy":
        beta, matching, unmatched_indices = _compute_beta_hopcroft_karp_scipy(inds)

    else:
        raise ValueError(
            "`algorithm` must be either `push_relabel_graph_tool`"
            " or `hopcroft_karp_scipy` or `edmonds_karp_graph_tool`."
        )

    return beta, matching, unmatched_indices


def _compute_beta(
    x: pd.DataFrame,
    y: pd.DataFrame,
    alpha: float,
    algorithm_candidates: str | Sequence[tuple[str, int]],
    column_weights: None | dict[str, float] = None,
    p: float = 1,
) -> tuple[float, float, str, float, tuple[tuple], tuple[tuple]]:
    assert (
        not isinstance(algorithm_candidates, str)
        and all(algorithm in ALGORITHMS for algorithm, _ in algorithm_candidates)
    ) or algorithm_candidates == "auto"

    LOGGER.info(f"Generating {alpha}-radius nearest neighbors")
    inds, density = _compute_radius_neighbors(x, y, alpha, p, column_weights)
    LOGGER.info(f"Done generating {alpha}-radius nearest neighbors")
    LOGGER.info(f"Algorithm candidates for ɑ={alpha}: {algorithm_candidates}")

    beta = float("nan")
    duration = float("nan")
    candidate = ""
    matching = ((), ())
    unmatched_indices = ((), ())

    if algorithm_candidates == "auto":
        algorithm_candidates = [("push_relabel_igraph", 60 * 60)]

        if (
            density < HOPCROFT_KARP_SCIPY_MAX_DENSITY_THRESHOLD
            and len(x) < HOPCROFT_KARP_SCIPY_MAX_SIZE_THRESHOLD
        ):
            algorithm_candidates.insert(0, ("hopcroft_karp_scipy", 5 * 60))

    for candidate, candidate_timeout in algorithm_candidates:
        logging_signature = (
            f"ɑ={alpha} E/½V²={density:.5f} {candidate} {candidate_timeout}"
        )

        # TODO: In hindsight, using `multiprocessing` would be the right thing
        # Then the process can be killed directly,
        # A queue is necessary for that
        # https://docs.python.org/3/library/multiprocessing.html
        with ProcessPoolExecutor(max_workers=1) as executor:
            pid = executor.submit(os.getpid).result()

            LOGGER.info(f"Submitting task ({pid}): {logging_signature}")
            future = executor.submit(_compute_beta_by_algorithm, inds, candidate)

            try:
                LOGGER.info(f"Waiting to task ({pid}): {logging_signature}")

                start = time.time()
                beta, matching, unmatched_indices = future.result(
                    timeout=candidate_timeout
                )
                end = time.time()
                duration = end - start

                matching = (
                    tuple(int(i) for i in matching[0]),
                    tuple(int(i) for i in matching[1]),
                )
                unmatched_indices = (
                    tuple(int(i) for i in unmatched_indices[0]),
                    tuple(int(i) for i in unmatched_indices[1]),
                )

            except Exception:
                LOGGER.info(f"Exception ({pid}): {logging_signature}", exc_info=True)

                # TODO: what to do if the processed is not killed?
                try:
                    os.kill(pid, signal.SIGILL)
                except OSError:
                    pass
                else:
                    LOGGER.info(f"Kill signal sent ({pid}): {logging_signature}")

            else:
                LOGGER.info(f"Success ({pid}): {beta} {density} {duration}")
                break

    return beta, density, candidate, duration, matching, unmatched_indices


def evaluate_faithfulness(
    x: pd.DataFrame,
    y: pd.DataFrame,
    column_weights: dict[str, float],
    alphas: Sequence[float],
    algorithm_candidates: str | Sequence[tuple[str, int]] = "auto",
) -> pd.DataFrame:
    """
    Evaluate faithfulness, i.e. proportion of not-matched pairs between two tables.

    Algorithms:
        1. `push_relabel_igraph`
            O(√VE), VERY fast implementation and good choice for *dense* graphs.
            From graph_tool (boost).
        2. `push_relabel_graph_tool`
            O(V^2√E), fast implementation and good choice for *dense* graphs.
            From graph_tool (boost).
        3. `hopcroft_karp_scipy`
            O(√VE), fast implementation and good choice for *sparse* graphs.
            From Scipy.
        4. `edmonds_karp_graph_tool`
            O(VE^2). From graph_tool (boost).

    In practice, the Push-Relable algorithm works better then Hopcroft-Karp algorithm .
    https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm#Comparison_with_other_bipartite_matching_algorithms

    The implementation of the Push-Relabel from the `igraph` package:
    https://github.com/igraph/igraph/blob/master/src/misc/matching.c

    Args:
        x (DataFrame): The first DataFrame.
        y (DataFrame): The second DataFrame.
        column_weights (dict, optional): The scaling weight (number or bins) per column.
        alphas (sequence): Sequence of the radious of the neighbors to try.
        algorithm_candidates (sequence): Tuple of (1) Which max cardinality
                                          bipartite matching algorithm to use;
                                          and (2)  timeout for running it.
    """

    faithfulness_records = []

    for alpha in tqdm(alphas):
        (
            beta,
            density,
            successful_candidate,
            duration,
            matching,
            unmatched_indices,
        ) = _compute_beta(x, y, alpha, algorithm_candidates, column_weights)

        faithfulness_records.append(
            {
                "ɑ": alpha,
                "β": beta,
                "E/½V²": density,
                "algorithm": successful_candidate,
                "duration": duration,
                "matching": matching,
                "unmatched_indices": unmatched_indices,
            }
        )

    return pd.DataFrame(faithfulness_records)


def find_assignment_matching(
    x: pd.DataFrame,
    y: pd.DataFrame,
    column_weights: dict[str, float],
    p: float = 1,
) -> tuple[tuple[ArrayLike, ArrayLike], ArrayLike]:
    x = scale_columns_by_weights(x, column_weights)
    y = scale_columns_by_weights(y, column_weights)

    dists = cdist(x, y, "minkowski", p=p)
    matching = linear_sum_assignment(dists)
    matching_costs = dists[matching]

    return matching, matching_costs
