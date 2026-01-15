import numpy as np
from scipy.stats import expon
from subprocess import call
import os

class TwoStateDiffusionSimulator:
    def __init__(self, delta, tracking_error, sim_time, particle, Tau, D, ratio, dt):
        """
        Initialize the simulation with parameters.
        """
        self.delta = delta                    # Segment length (steps)
        self.tracking_error = tracking_error  # Localization error (noise)
        self.sim_time = sim_time              # Number of segments
        self.particle = particle              # Particle identifier (used for filenames)
        self.tau = Tau                        # Average state duration
        self.D_0 = D                          # Diffusion coefficient for state 0
        self.D_1 = D * ratio                  # Diffusion coefficient for state 1
        self.dt = dt                          # Time step (not used directly, may be useful for future)
        self.total_steps = self.delta * self.sim_time

    def compute_trajectory_2_state(self):
        """
        Generate a 2-state trajectory with alternating diffusion coefficients.
        """
        x, y = [0], [0]          # Initial positions
        total_barcode = []       # Encodes which state the particle is in at each step
        D = self.D_0             # Start in state 0
        steps_generated = 0

        while steps_generated < self.total_steps:
            t = int(expon.rvs(scale=self.tau * self.delta))  # Duration in current state

            # Generate displacements for the current state
            dx = np.random.normal(0, np.sqrt(2 * D), t)
            dy = np.random.normal(0, np.sqrt(2 * D), t)

            # Append displacements
            x.extend(dx)
            y.extend(dy)

            # Add barcode (0 for D0, 1 for D1)
            total_barcode.extend([0 if D == self.D_0 else 1] * t)

            # Toggle state
            D = self.D_1 if D == self.D_0 else self.D_0

            steps_generated += t

        # Cumulative sum to get position
        x = np.cumsum(x)[:self.total_steps]
        y = np.cumsum(y)[:self.total_steps]
        barcode = total_barcode[:self.total_steps - 1]

        return np.array(x), np.array(y), np.array(barcode)

    def apply_blur(self, x, y, barcode):
        """
        Blur the trajectory by averaging over each segment.
        """
        blurred_x, blurred_y, blurred_barcode = [], [], []

        for i in range(self.sim_time):
            start, end = i * self.delta, (i + 1) * self.delta
            segment_x = x[start:end]
            segment_y = y[start:end]
            segment_barcode = barcode[start:end - 1]

            # Decide the barcode based on the average value
            label = 1 if np.mean(segment_barcode) >= 0.5 else 0

            blurred_x.append(np.mean(segment_x))
            blurred_y.append(np.mean(segment_y))
            blurred_barcode.append(label)

        # Remove the last barcode to match length
        return np.array(blurred_x), np.array(blurred_y), np.array(blurred_barcode[:-1])

    def run(self):
        """
        Run the full simulation: generate, blur, add noise, and save.
        """
        # Step 1: Simulate trajectory
        x, y, barcode = self.compute_trajectory_2_state()

        # Step 2: Blur the data if segment length > 1
        if self.delta != 1:
            blurred_x, blurred_y, blurred_barcode = self.apply_blur(x, y, barcode)
        else:
            blurred_x, blurred_y, blurred_barcode = x, y, barcode

        # Step 3: Add localization noise
        blurred_x += np.random.normal(0, self.tracking_error, size=blurred_x.shape)
        blurred_y += np.random.normal(0, self.tracking_error, size=blurred_y.shape)

        # Step 4: Compute displacements
        dx = np.diff(blurred_x)
        dy = np.diff(blurred_y)
        displacements = np.sqrt(dx**2 + dy**2)

        # Step 5: Save results
        self.save_data(displacements, blurred_barcode, blurred_x, blurred_y)

    def save_data(self, displacements, blurred_barcode, blurred_x, blurred_y):
        """
        Save simulated displacement, trajectory, and ground truth data to disk.
        """
        path = f'raw-data/sim_time_{self.sim_time}/D_{self.D_0}_ratio_{self.D_1/self.D_0}_Tau_{self.tau}'
        full_path = f'{path}/blurring_{self.delta}/err_{self.tracking_error}'

        os.makedirs(full_path, exist_ok=True)

        # Save displacements
        np.savetxt(f'{full_path}/diff_{self.particle}.dat',
                   np.transpose([np.arange(0, self.sim_time - 1), displacements]))

        # Save ground truth (state labels)
        np.savetxt(f'{full_path}/ground_truth_{self.particle}.dat',
                   np.transpose([blurred_barcode]))

        # Save trajectory (x, y)
        np.savetxt(f'{full_path}/trajectory_{self.particle}.dat',
                   np.transpose([blurred_x, blurred_y]))
