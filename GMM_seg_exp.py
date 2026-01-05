"""
GMM-based segmentation of experimental trajectory data
======================================================

This script:
- Loads experimental particle tracking data from a CSV or Excel file
- Handles trajectories with unequal lengths
- Computes stepwise displacements
- Applies Gaussian filtering
- Fits a Gaussian Mixture Model (GMM)
- Optimizes the filter size by minimizing overlap between GMM components
- Produces a discrete "barcode" segmentation for each trajectory
"""

import os
import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


class GMMFilterSegmentation:
    """
    Main class for GMM-based segmentation of experimental trajectory data.
    """

    def __init__(
        self,
        data_path,
        nb_states=2,
        filter_bounds=(0.01, 10.0),
        random_state=0,
        output_dir="output",
    ):
        """
        Parameters
        ----------
        data_path : str
            Path to CSV or Excel file containing experimental trajectories.

        nb_states : int
            Number of Gaussian states in the GMM.

        filter_bounds : tuple(float, float)
            Lower and upper bounds for Gaussian filter size optimization.

        random_state : int
            Random seed for reproducible GMM fitting.

        output_dir : str
            Directory where barcode files will be saved.
        """

        self.data_path = data_path
        self.nb_states = nb_states
        self.filter_bounds = filter_bounds
        self.random_state = random_state
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Load and prepare displacement data
        self.data_total = self._load_data()
        self.n_tracks = self.data_total.shape[1]

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------

    def _load_data(self):
        """
        Load experimental tracking data and convert it into a padded
        displacement matrix of shape (T_max, n_tracks).

        Expected columns in the data file:
        - TRACK_ID
        - FRAME
        - POSITION_X
        - POSITION_Y
        """

        # Load file depending on extension
        if self.data_path.endswith(".xlsx"):
            df = pd.read_excel(self.data_path)
        else:
            df = pd.read_csv(self.data_path)

        # Ensure required columns exist
        required_cols = ["TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert to numeric and clean
        for col in ["FRAME", "POSITION_X", "POSITION_Y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required_cols)

        tracks = []

        # Process each trajectory independently
        for track_id, subdf in df.groupby("TRACK_ID"):
            subdf = subdf.sort_values("FRAME")

            x = subdf["POSITION_X"].values
            y = subdf["POSITION_Y"].values

            if len(x) < 5:
                continue  # cannot compute displacement

            # Stepwise displacement magnitude
            dx = np.diff(x)
            dy = np.diff(y)
            disp = np.sqrt(dx**2 + dy**2)

            tracks.append(disp)

        if len(tracks) == 0:
            raise RuntimeError("No valid trajectories found.")

        # Pad tracks to same length using NaNs
        max_len = max(len(t) for t in tracks)
        data_total = np.full((max_len, len(tracks)), np.nan)

        for i, t in enumerate(tracks):
            data_total[: len(t), i] = t

        return data_total

    # ------------------------------------------------------------------
    # GMM FITTING
    # ------------------------------------------------------------------

    def _fit_gmm(self, data):
        """
        Fit a Gaussian Mixture Model to 1D displacement data.
        """

        # Remove NaNs before fitting
        data = data[~np.isnan(data)]

        gm = GaussianMixture(
            n_components=self.nb_states,
            tol=1e-6,
            random_state=self.random_state,
        ).fit(data.reshape(-1, 1))

        A = gm.weights_
        mu = gm.means_.flatten()
        sig = np.sqrt(gm.covariances_[:, 0, 0])

        return A, mu, sig

    # ------------------------------------------------------------------
    # OVERLAP & SEPARATORS
    # ------------------------------------------------------------------

    def _separator(self, A1, mu1, sig1, A2, mu2, sig2, max_val):
        """
        Compute intersection point between two weighted Gaussians.
        """

        a = sig1**2 - sig2**2
        b = 2 * (mu1 * sig2**2 - mu2 * sig1**2)
        c = (
            mu2**2 * sig1**2
            - mu1**2 * sig2**2
            - 2 * sig1**2 * sig2**2 * np.log((sig1 * A2) / (sig2 * A1))
        )

        disc = b**2 - 4 * a * c
        if disc < 0:
            return np.inf

        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        for x in roots:
            if 0 <= x <= max_val:
                return x

        return np.inf

    def _numerical_overlap(self, sep, A1, mu1, sig1, A2, mu2, sig2):
        """
        Compute numerical overlap area between two Gaussians.
        """

        if sep == np.inf:
            return np.inf

        x1 = np.linspace(0, sep, 1000)
        x2 = np.linspace(sep, sep + 10 * sig2, 1000)

        G1 = np.sum(A1 * norm.pdf(x2, mu1, sig1)) * (x2[1] - x2[0])
        G2 = np.sum(A2 * norm.pdf(x1, mu2, sig2)) * (x1[1] - x1[0])

        return G1 + G2

    def _total_overlap(self, A, mu, sig, max_val):
        """
        Compute total overlap between all GMM components.
        """

        idx = np.argsort(mu)
        A, mu, sig = A[idx], mu[idx], sig[idx]

        total_overlap = 0.0
        separators = []

        for i in range(len(A) - 1):
            for j in range(i + 1, len(A)):
                sep = self._separator(A[i], mu[i], sig[i], A[j], mu[j], sig[j], max_val)
                total_overlap += self._numerical_overlap(
                    sep, A[i], mu[i], sig[i], A[j], mu[j], sig[j]
                )

        for i in range(len(A) - 1):
            sep = self._separator(
                A[i], mu[i], sig[i], A[i + 1], mu[i + 1], sig[i + 1], max_val
            )
            separators.append(sep)

        return total_overlap, np.array(separators)

    # ------------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------------

    def _filter_optimize(self, filter_size):
        """
        Apply Gaussian filtering and compute total GMM overlap.
        """

        filtered = gaussian_filter(self.data_total, sigma=(0, filter_size))
        max_val = np.nanmax(filtered)

        A, mu, sig = self._fit_gmm(filtered.flatten())
        overlap, separators = self._total_overlap(A, mu, sig, max_val)

        return overlap, np.sort(separators)

    def _objective(self, filter_size):
        """
        Objective function minimized during optimization.
        """

        overlap, _ = self._filter_optimize(filter_size)
        return overlap

    # ------------------------------------------------------------------
    # BARCODE PREDICTION
    # ------------------------------------------------------------------

    def _barcode_predict(self, data_column, separators):
        """
        Convert a filtered trajectory into a discrete barcode.
        """

        code = np.full(len(data_column), len(separators))

        for i, sep in enumerate(separators):
            below = data_column < sep
            code[below] = np.minimum(code[below], i)

        return code

    def compute_diffusion_coefficients(self, dt=1.0):
        """
        Compute diffusion coefficients for each GMM state using
        segmented trajectories.

        Parameters
        ----------
        dt : float
            Time step between consecutive frames (default: 1.0).

        Returns
        -------
        D_states : np.ndarray
            Diffusion coefficient for each state (length = nb_states).
        """

        # Ensure segmentation has been run
        if not hasattr(self, "optimal_separators"):
            raise RuntimeError(
                "Segmentation must be run before computing diffusion coefficients."
            )

        # Accumulate squared displacements per state
        sq_disp_per_state = [[] for _ in range(self.nb_states)]

        for i in range(self.n_tracks):
            data = self.data_total[:, i]
            valid = ~np.isnan(data)

            if np.sum(valid) == 0:
                continue

            # Recompute filtered signal and barcode
            filtered = gaussian_filter(data[valid], sigma=self.optimal_filter)
            barcode = self._barcode_predict(filtered, self.optimal_separators)

            # Assign squared displacements to states
            for state in range(self.nb_states):
                mask = barcode == state
                sq_disp = data[valid][mask] ** 2
                sq_disp_per_state[state].extend(sq_disp.tolist())

        # Compute diffusion coefficients
        D_states = np.zeros(self.nb_states)

        for s in range(self.nb_states):
            if len(sq_disp_per_state[s]) == 0:
                D_states[s] = np.nan
            else:
                mean_sq_disp = np.mean(sq_disp_per_state[s])
                D_states[s] = (3/2) * mean_sq_disp / (4.0 * dt) # Berglund correction

        return D_states

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def run(self):
        """
        Run full optimization and segmentation pipeline.
        """

        result = minimize_scalar(
            self._objective,
            bounds=self.filter_bounds,
            method="bounded",
        )

        if not result.success:
            raise RuntimeError("Filter optimization failed.")

        optimal_filter = result.x
        _, separators = self._filter_optimize(optimal_filter)

        print(f"Optimal filter size: {optimal_filter:.4f}")
        print(f"Separators: {separators}")

        # Segment each trajectory independently
        for i in range(self.n_tracks):
            data = self.data_total[:, i]
            valid = ~np.isnan(data)

            filtered = np.full_like(data, np.nan)
            filtered[valid] = gaussian_filter(data[valid], sigma=optimal_filter)

            barcode = self._barcode_predict(filtered[valid], separators)

            np.savetxt(
                os.path.join(self.output_dir, f"barcode_track_{i}.dat"),
                barcode,
            )
            
        self.optimal_filter = optimal_filter
        self.optimal_separators = separators


# ----------------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------------

if __name__ == "__main__":

    segmenter = GMMFilterSegmentation(
        data_path="data.csv",   # or .xlsx
        nb_states=2,
        output_dir="barcodes",
    )

    segmenter.run()

    D_states = segmenter.compute_diffusion_coefficients(dt=30e-3)

    for i, D in enumerate(D_states):
        print(f"State {i}: D = {D:.4e} um^2/s")