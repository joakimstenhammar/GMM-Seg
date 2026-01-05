import numpy as np
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


class GMMFilterSegmentation:
    def __init__(self, nb_states, D, ratio, Tau, delta, tracking_error, n, sim_time):
        self.nb_states = nb_states
        self.D = D
        self.ratio = ratio
        self.Tau = Tau
        self.delta = delta
        self.tracking_error = tracking_error
        self.n = n
        self.sim_time = sim_time

        self.path = f'raw-data/sim_time_{sim_time}/D_{D}_ratio_{ratio}_Tau_{Tau}/blurring_{delta}/err_{tracking_error}'
        self.data_total = self._load_data()

    def _load_data(self):
        """
        Load all displacement data files into a 2D array.
        """
        data_total = []
        for i in range(self.n):
            data_i = np.loadtxt(f'{self.path}/diff_{i}.dat').reshape(-1, 2)[:, 1]
            data_total.append(data_i)
        return np.column_stack(data_total)

    def _fit_gmm(self, data):
        """
        Fit a Gaussian Mixture Model to the flattened data.
        """
        gm = GaussianMixture(n_components=self.nb_states, tol=1e-6, random_state=0).fit(data.reshape(-1, 1))
        A = gm.weights_
        mu = gm.means_.flatten()
        sig = np.sqrt(gm.covariances_[:, 0, 0])
        return A, mu, sig

    def _separator(self, A1, mu1, sig1, A2, mu2, sig2, max_val):
        """
        Compute the point that separates two Gaussian distributions.
        """
        a = sig1**2 - sig2**2
        b = 2 * (mu1 * sig2**2 - mu2 * sig1**2)
        c = mu2**2 * sig1**2 - mu1**2 * sig2**2 - 2 * sig1**2 * sig2**2 * np.log((sig1 * A2) / (sig2 * A1))

        disc = b**2 - 4 * a * c
        if disc < 0:
            return np.inf

        x1 = (-b + np.sqrt(disc)) / (2 * a)
        x2 = (-b - np.sqrt(disc)) / (2 * a)

        for x in [x1, x2]:
            if 0 <= x <= max_val:
                return x
        return np.inf

    def _numerical_overlap(self, sep, A1, mu1, sig1, A2, mu2, sig2):
        """
        Compute overlap area of two Gaussians numerically.
        """
        if sep is None or sep == np.inf:
            return np.inf

        x1 = np.linspace(0, sep, 1000)
        x2 = np.linspace(sep, sep + 10 * sig2, 1000)

        G1 = np.sum(A1 * norm.pdf(x2, mu1, sig1)) * (x2[1] - x2[0])
        G2 = np.sum(A2 * norm.pdf(x1, mu2, sig2)) * (x1[1] - x1[0])

        return G1 + G2

    def _total_overlap(self, A, mu, sig, max_val):
        """
        Compute total overlap and separators between GMM components.
        """
        idx = np.argsort(mu)
        A, mu, sig = A[idx], mu[idx], sig[idx]

        total = 0
        separators = []

        for i in range(len(A) - 1):
            for j in range(i + 1, len(A)):
                sep = self._separator(A[i], mu[i], sig[i], A[j], mu[j], sig[j], max_val)
                total += self._numerical_overlap(sep, A[i], mu[i], sig[i], A[j], mu[j], sig[j])

        for i in range(len(A) - 1):
            sep = self._separator(A[i], mu[i], sig[i], A[i + 1], mu[i + 1], sig[i + 1], max_val)
            separators.append(sep)

        return total, np.array(separators)

    def _filter_optimize(self, filter_size):
        """
        Apply Gaussian filter and return overlap + separators.
        """
        filtered = gaussian_filter(self.data_total, sigma=(0, filter_size))
        max_val = np.max(filtered)
        A, mu, sig = self._fit_gmm(filtered)
        overlap, separators = self._total_overlap(A, mu, sig, max_val)
        return overlap, np.sort(separators)

    def _overlap_objective(self, filter_size):
        """
        Objective function to minimize: total GMM overlap for a given filter size.
        """
        overlap, _ = self._filter_optimize(filter_size)
        return overlap

    def _barcode_predict(self, data_column, separators):
        """
        Convert filtered data into barcode using thresholds.
        """
        code = np.full(len(data_column), len(separators))
        for i, sep in enumerate(separators):
            below = data_column < sep
            code[below] = np.minimum(code[below], i)
        return code

    def _task(self, index, filter_size):
        """
        Parallelizable task for a given filter size.
        """
        overlap, separators = self._filter_optimize(filter_size)
        return index, overlap, separators

    def run(self):
        """
        Optimize Gaussian filter size using scalar minimization (Brent’s method),
        then segment each trajectory and save predicted barcodes.
        """
        result = minimize_scalar(self._overlap_objective, bounds=(0.01, 10.0), method='bounded')

        if not result.success:
            raise RuntimeError("Optimization failed.")

        optimal_filter = result.x
        _, optimal_separators = self._filter_optimize(optimal_filter)

        print(f"Optimal filter size: {optimal_filter:.4f}")
        print(f"Separators: {optimal_separators}")

        for i in range(self.n):
            data_filtered = gaussian_filter(self.data_total[:, i], sigma=optimal_filter)
            barcode = self._barcode_predict(data_filtered, optimal_separators)
            np.savetxt(f'{self.path}/barcode_{i}.dat', barcode)

    def compute_accuracy(self, n):
        """
        Compare predicted barcode against ground truth for column n.
        """
        path = self.path
        gt_path = f'{path}/ground_truth_{n}.dat'
        pred_path = f'{path}/barcode_{n}.dat'

        if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
            print(f"Missing files for n = {n}")
            return None

        ground_truth = np.loadtxt(gt_path)
        predicted = np.loadtxt(pred_path).reshape(-1)

        if len(ground_truth) != len(predicted):
            print(f"Length mismatch for n = {n}")
            return None

        accuracy = np.sum(ground_truth == predicted) / len(ground_truth)
        print(f"n = {n} | Accuracy: {accuracy:.4f}")
        return accuracy

# Input parameters -- To localize synthetic data

nb_states        = 2
n                = 1
sim_time         = 2000
Tau              = 40
D                = 1
ratio            = 10.0
dt               = 1

delta            = 1
tracking_error   = 0.0

X = (nb_states, D, ratio, Tau, delta, tracking_error, n, sim_time)
segmenter = GMMFilterSegmentation(*X)
segmenter.run()

mean_acc = 0.0
for i in range(n):
    mean_acc += segmenter.compute_accuracy(i)/n
    
print(f"Mean accuracy over {n} particles: {mean_acc:.4f}")