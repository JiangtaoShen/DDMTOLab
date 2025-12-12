import numpy as np
from scipy.spatial.distance import cdist


class IGD:
    """
    Inverted Generational Distance (IGD) metric.
    Lower is better.
    """

    def __init__(self):
        self.name = "IGD"
        self.sign = -1

    def calculate(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """
        Compute IGD(objs, pf)

        Parameters
        ----------
        objs : (n, m) obtained objective vectors
        pf   : (n_pf, m) true Pareto front

        Returns
        -------
        float : IGD value
        """
        objs = np.asarray(objs)
        pf = np.asarray(pf)

        # basic check
        if objs.size == 0 or pf.size == 0:
            return np.nan
        if objs.ndim != 2 or pf.ndim != 2:
            return np.nan
        if objs.shape[1] != pf.shape[1]:
            return np.nan

        # pairwise distances (n_pf x n)
        distances = cdist(pf, objs, metric='euclidean')

        # IGD = average of nearest distances (for each PF point)
        return float(np.mean(np.min(distances, axis=1)))

    def __call__(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """Allow instance to be called like a function"""
        return self.calculate(objs, pf)


class GD:
    """
    Generational Distance (GD) metric.
    Lower is better.
    """

    def __init__(self):
        self.name = "GD"
        self.sign = -1

    def calculate(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """
        Compute GD(objs, pf)

        Parameters
        ----------
        objs : (n, m) obtained objective vectors
        pf   : (n_pf, m) true Pareto front

        Returns
        -------
        float : GD value
        """
        objs = np.asarray(objs)
        pf = np.asarray(pf)

        # basic check
        if objs.size == 0 or pf.size == 0:
            return np.nan
        if objs.ndim != 2 or pf.ndim != 2:
            return np.nan
        if objs.shape[1] != pf.shape[1]:
            return np.nan

        # pairwise distances (n x n_pf)
        distances = cdist(objs, pf, metric='euclidean')

        # GD = norm of nearest distances (for each obtained point) / number of points
        min_distances = np.min(distances, axis=1)
        return float(np.linalg.norm(min_distances) / len(min_distances))

    def __call__(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """Allow instance to be called like a function"""
        return self.calculate(objs, pf)


class IGDp:
    """
    Inverted Generational Distance Plus (IGD+) metric.
    Lower is better.
    """

    def __init__(self):
        self.name = "IGDp"
        self.sign = -1

    def calculate(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """
        Compute IGD+(objs, pf)

        Parameters
        ----------
        objs : (n, m) obtained objective vectors
        pf   : (n_pf, m) true Pareto front

        Returns
        -------
        float : IGD+ value
        """
        objs = np.asarray(objs)
        pf = np.asarray(pf)

        # basic check
        if objs.size == 0 or pf.size == 0:
            return np.nan
        if objs.ndim != 2 or pf.ndim != 2:
            return np.nan
        if objs.shape[1] != pf.shape[1]:
            return np.nan

        n_pf, m = pf.shape
        n = objs.shape[0]

        delta = np.zeros(n_pf)

        for i in range(n_pf):
            # For each reference point, compute modified distance to all obtained points
            # max(objs - pf[i], 0): only count positive differences (dominated components)
            diff = objs - pf[i]  # (n, m)
            diff = np.maximum(diff, 0)  # (n, m)

            # Euclidean distance for each obtained point
            distances = np.sqrt(np.sum(diff ** 2, axis=1))  # (n,)

            # Minimum distance to this reference point
            delta[i] = np.min(distances)

        # IGD+ = average of minimum modified distances
        return float(np.mean(delta))

    def __call__(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """Allow instance to be called like a function"""
        return self.calculate(objs, pf)


class HV:
    """
    Hypervolume (HV) metric.
    Higher is better.
    """

    def __init__(self):
        self.name = "HV"
        self.sign = 1

    def calculate(self, objs: np.ndarray, pf: np.ndarray = None, reference: np.ndarray = None) -> float:
        """
        Compute HV for a set of objective vectors.

        Parameters
        ----------
        objs : np.ndarray
            Objective matrix, shape (n, m)
        pf : np.ndarray, optional
            True Pareto front for normalization, shape (n_pf, m)
        reference : np.ndarray, optional
            Reference point for HV calculation, shape (m,)

        Returns
        -------
        float
            HV value
        """
        if pf is None and reference is None:
            raise ValueError("Either pf or reference must be provided")

        objs = np.array(objs)
        if objs.ndim != 2 or objs.shape[0] == 0:
            return 0.0

        n, m = objs.shape

        # Normalization
        if pf is not None:
            fmin = np.minimum(np.min(objs, axis=0), np.zeros(m))
            fmax = np.max(pf, axis=0)
            objs_norm = (objs - fmin) / ((fmax - fmin) * 1.1)
            valid_mask = np.all(objs_norm <= 1, axis=1)
            objs_filtered = objs_norm[valid_mask]
            ref_point_norm = np.ones(m)
        else:
            fmin = np.minimum(np.min(objs, axis=0), np.zeros(m))
            fmax = reference
            objs_norm = (objs - fmin) / ((fmax - fmin) * 1.1)
            valid_mask = np.all(objs_norm <= 1, axis=1)
            objs_filtered = objs_norm[valid_mask]
            ref_point_norm = np.ones(m) / 1.1

        if objs_filtered.shape[0] == 0:
            return 0.0

        if m < 4:
            return self._exact_hv(objs_filtered, ref_point_norm)
        else:
            return self._monte_carlo_hv(objs_filtered, ref_point_norm)

    def __call__(self, objs: np.ndarray, pf: np.ndarray = None, reference: np.ndarray = None) -> float:
        """Allow instance to be called like a function."""
        return self.calculate(objs, pf, reference)

    def _exact_hv(self, objs: np.ndarray, ref_point: np.ndarray) -> float:
        n, m = objs.shape
        if m == 2:
            sorted_objs = objs[np.argsort(objs[:, 0])]
            hv = 0.0
            prev_x = sorted_objs[0, 0]
            min_y = sorted_objs[0, 1]
            for i in range(n):
                x, y = sorted_objs[i]
                if i > 0:
                    hv += (x - prev_x) * (ref_point[1] - min_y)
                    prev_x = x
                if y < min_y:
                    min_y = y
            hv += (ref_point[0] - sorted_objs[-1, 0]) * (ref_point[1] - min_y)
            return hv
        elif m == 3:
            pl = objs[np.argsort(objs[:, 0])]
            S = [(1.0, pl)]
            for k in range(m - 1):
                S_new = []
                for volume, points in S:
                    slices = self._slice(points, k, ref_point)
                    for slice_volume, slice_points in slices:
                        S_new = self._add((volume * slice_volume, slice_points), S_new)
                S = S_new
            hv = 0.0
            for volume, points in S:
                if len(points) > 0:
                    p = points[0]
                    hv += volume * abs(p[m - 1] - ref_point[m - 1])
            return hv
        else:
            raise ValueError(f"Exact HV only supports m < 4, got m={m}")

    def _slice(self, pl: np.ndarray, k: int, ref_point: np.ndarray) -> list:
        if len(pl) == 0:
            return []
        S = []
        p = pl[0]
        pl = pl[1:]
        ql = np.empty((0, pl.shape[1] if len(pl) > 0 else len(p)))
        while len(pl) > 0:
            ql = self._insert(p, k + 1, ql)
            p_next = pl[0]
            volume = abs(p[k] - p_next[k])
            S = self._add((volume, ql.copy()), S)
            p = p_next
            pl = pl[1:]
        ql = self._insert(p, k + 1, ql)
        volume = abs(p[k] - ref_point[k])
        S = self._add((volume, ql.copy()), S)
        return S

    def _insert(self, p: np.ndarray, k: int, pl: np.ndarray) -> np.ndarray:
        m = len(p)
        if len(pl) == 0:
            return np.array([p])
        ql = []
        inserted = False
        for q in pl:
            if not inserted and q[k] >= p[k]:
                ql.append(p)
                inserted = True
            flag1 = flag2 = False
            for j in range(k, m):
                if p[j] < q[j]:
                    flag1 = True
                elif p[j] > q[j]:
                    flag2 = True
            if not (flag1 and not flag2):
                ql.append(q)
        if not inserted:
            ql.append(p)
        return np.array(ql) if ql else np.empty((0, m))

    def _add(self, cell: tuple, S: list) -> list:
        volume, points = cell
        for i, (v, p) in enumerate(S):
            if len(points) == len(p) and np.allclose(points, p):
                S[i] = (v + volume, p)
                return S
        S.append(cell)
        return S

    def _monte_carlo_hv(self, objs: np.ndarray, ref_point: np.ndarray) -> float:
        n_samples = int(1e6)
        m = objs.shape[1]
        min_vals = np.min(objs, axis=0)
        max_vals = ref_point
        samples = np.random.uniform(low=min_vals, high=max_vals, size=(n_samples, m))
        dominated = np.zeros(n_samples, dtype=bool)
        for i in range(objs.shape[0]):
            dominated |= np.all(samples >= objs[i], axis=1)
        volume_box = np.prod(max_vals - min_vals)
        return volume_box * (np.sum(dominated) / n_samples)


class FR:
    """
    Feasible Rate metric.
    Calculates the proportion of feasible solutions in the population.
    Higher is better (more feasible solutions).
    """

    def __init__(self):
        self.name = "Feasible_rate"
        self.sign = 1  # Higher is better

    def calculate(self, cons: np.ndarray) -> float:
        """
        Compute feasible rate

        Parameters
        ----------
        cons : (n, c) constraint violation matrix
               where n is the number of solutions
               and c is the number of constraints
               A solution is feasible if all constraints <= 0

        Returns
        -------
        float : Feasible rate (proportion of feasible solutions)
        """
        cons = np.asarray(cons)

        # basic check
        if cons.size == 0:
            return np.nan

        # Handle 1D array (single constraint)
        if cons.ndim == 1:
            cons = cons.reshape(-1, 1)

        if cons.ndim != 2:
            return np.nan

        # Check if all constraints are satisfied (<=0) for each solution
        feasible = np.all(cons <= 0, axis=1)  # (n,) boolean array

        # Calculate proportion of feasible solutions
        return float(np.mean(feasible))

    def __call__(self, cons: np.ndarray) -> float:
        """Allow instance to be called like a function"""
        return self.calculate(cons)


class CV:
    """
    Constraint Violation (CV) metric.
    Lower is better (ideally 0 for feasible solutions).
    """

    def __init__(self):
        self.name = "CV"
        self.sign = -1  # Lower is better

    def calculate(self, cons: np.ndarray) -> float:
        """
        Compute CV metric - returns the best (minimum) CV in the population

        Parameters
        ----------
        cons : (n, c) constraint violation matrix
               where n is the number of solutions
               and c is the number of constraints
               Constraint is satisfied when cons <= 0

        Returns
        -------
        float : CV value of the best solution (minimum CV)
        """
        cons = np.asarray(cons)

        # basic check
        if cons.size == 0:
            return np.nan

        # Handle 1D array (single solution with multiple constraints)
        if cons.ndim == 1:
            return float(np.sum(np.maximum(cons, 0)))

        if cons.ndim != 2:
            return np.nan

        # Calculate CV for each solution
        # CV = sum of constraint violations (only positive values count)
        cv_values = np.sum(np.maximum(cons, 0), axis=1)  # (n,)

        # Return the best (minimum) CV
        return float(np.min(cv_values))

    def __call__(self, cons: np.ndarray) -> float:
        """Allow instance to be called like a function"""
        return self.calculate(cons)


class DeltaP:
    """
    Averaged Hausdorff Distance (Δp) metric.
    Lower is better.
    """

    def __init__(self):
        self.name = "DeltaP"
        self.sign = -1  # Lower is better

    def calculate(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """
        Compute Δp(objs, pf)

        Parameters
        ----------
        objs : (n, m) obtained objective vectors
        pf   : (n_pf, m) true Pareto front

        Returns
        -------
        float : Δp value (Averaged Hausdorff Distance)
        """
        objs = np.asarray(objs)
        pf = np.asarray(pf)

        # basic check
        if objs.size == 0 or pf.size == 0:
            return np.nan
        if objs.ndim != 2 or pf.ndim != 2:
            return np.nan
        if objs.shape[1] != pf.shape[1]:
            return np.nan

        # pairwise distances (n_pf x n)
        distances = cdist(pf, objs, metric='euclidean')

        # GD component: mean of minimum distances from PF to objs
        gd = np.mean(np.min(distances, axis=1))

        # IGD component: mean of minimum distances from objs to PF
        igd = np.mean(np.min(distances, axis=0))

        # Δp = max(GD, IGD)
        return float(max(gd, igd))

    def __call__(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """Allow instance to be called like a function"""
        return self.calculate(objs, pf)


class Spacing:
    """
    Spacing metric.
    Lower is better.
    """

    def __init__(self):
        self.name = "Spacing"
        self.sign = -1  # Lower is better

    def calculate(self, objs: np.ndarray) -> float:
        """
        Compute Spacing metric

        Parameters
        ----------
        objs : (n, m) obtained objective vectors
               where n is the number of solutions
               and m is the number of objectives

        Returns
        -------
        float : Spacing value (standard deviation of nearest neighbor distances)
        """
        objs = np.asarray(objs)

        # basic check
        if objs.size == 0:
            return np.nan
        if objs.ndim != 2:
            return np.nan

        n = objs.shape[0]

        # Need at least 2 solutions to compute spacing
        if n < 2:
            return np.nan

        # Pairwise Manhattan (cityblock) distances (n x n)
        distances = cdist(objs, objs, metric='cityblock')

        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # Find minimum distance for each solution (nearest neighbor)
        min_distances = np.min(distances, axis=1)  # (n,)

        # Spacing = standard deviation of nearest neighbor distances
        return float(np.std(min_distances, ddof=0))

    def __call__(self, objs: np.ndarray) -> float:
        """Allow instance to be called like a function"""
        return self.calculate(objs)


class Spread:
    """
    Spread metric.
    Lower is better.
    """

    def __init__(self):
        self.name = "Spread"
        self.sign = -1  # Lower is better

    def calculate(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """
        Compute Spread metric

        Parameters
        ----------
        objs : (n, m) obtained objective vectors
               where n is the number of solutions
               and m is the number of objectives
        pf   : (n_pf, m) true Pareto front

        Returns
        -------
        float : Spread value
        """
        objs = np.asarray(objs)
        pf = np.asarray(pf)

        # basic check
        if objs.size == 0 or pf.size == 0:
            return np.nan
        if objs.ndim != 2 or pf.ndim != 2:
            return np.nan
        if objs.shape[1] != pf.shape[1]:
            return np.nan

        n, m = objs.shape

        # Need at least m solutions (number of objectives)
        if n < m:
            return np.nan

        # Pairwise distances between obtained solutions (n x n)
        dis1 = cdist(objs, objs, metric='euclidean')

        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(dis1, np.inf)

        # Find extreme points in the Pareto front
        # E contains indices of maximum values for each objective
        E = np.argmax(pf, axis=0)  # (m,)

        # Extract extreme points
        extreme_points = pf[E, :]  # (m, m)

        # Distances from extreme points to obtained solutions (m x n)
        dis2 = cdist(extreme_points, objs, metric='euclidean')

        # d1: sum of minimum distances from extreme points to obtained solutions
        d1 = np.sum(np.min(dis2, axis=1))

        # Minimum distances for each obtained solution to its nearest neighbor
        min_dis1 = np.min(dis1, axis=1)  # (n,)

        # d2: mean of nearest neighbor distances
        d2 = np.mean(min_dis1)

        # Spread formula
        numerator = d1 + np.sum(np.abs(min_dis1 - d2))
        denominator = d1 + (n - m) * d2

        # Avoid division by zero
        if denominator == 0:
            return np.nan

        score = numerator / denominator

        return float(score)

    def __call__(self, objs: np.ndarray, pf: np.ndarray) -> float:
        """Allow instance to be called like a function"""
        return self.calculate(objs, pf)
