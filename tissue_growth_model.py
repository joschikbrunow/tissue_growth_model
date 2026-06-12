import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.colors import LinearSegmentedColormap
import pickle
import os
from skimage.measure import find_contours


# Class for the simulation
class TissueSim2D:
    """
        TissueSim2D

        Class for the 2D-simulation of the dynamic growth of two competing cell populations
        (n1 and n2) based on a continuous model.
        The model takes growth, competition, pressure diffussion and self-diffusion into account.

        Usage:
            sim = TissueSim2D()  # Standard parameters
            sim = TissueSim2D(grid_size=100, G1=1.5, G2=0.8,...)  # Adjusted parameters
            sim.run()  # Run simulation
            sim.visualize()  # Visualize results
            sim.save("name") # Save simulation
            sim.load("name") # Load simulation

        Parameters:
            - Spatial: grid_size (number of divisions of [0,1]^2)
            - Time: count_timesteps (number of time steps), dt (length of a time step), snapscore (storage rate)
            - Physical: beta1, beta2 (pressure sensitivity), eps1, eps2 (self-diffusion), bound_cond (boundary condition)
            - Biological: G1, G2 (growth rate), comp_koef1, comp_koef2 (competition)
            - Initialization: center1, center2, (Coordinates of initial populations) peak1, peak2 (peak initial density), sigma1, sigma2 (variance), background1, background2 (additional initial density on grid)
        """
    
    # Class constructor
    def __init__(self,
                 # Gridsize -> gridsize^2 control-volumes
                 grid_size = 50,
                 # Number of time steps to simulate
                 count_timesteps = 1000000,
                 # Length of a time step
                 dt = 5e-6,
                 # Every [snapscore] time step is saved
                 snapscore = 100,
                 # Boundary condition ("neumann", "dirichlet")
                 bound_cond="neumann",
                 # Growth rate of n1
                 G1 = 1.0,
                 # Growth rate of n2
                 G2 = 1.0,
                 # Growth inhibition of n1 through n2 (Ability to compete of n1)
                 comp_koef1=1.0,
                 # Growth inhibition of n2 through n1 (Ability to compete of n2)
                 comp_koef2 = 1.0,
                 # Pressure sensitivity of n1 (Higher beta -> More movement due to pressure)
                 beta1=1.0,
                 # Pressure sensitivity of n2 (Higher beta -> More movement due to pressure)
                 beta2=1.0,
                 # Self-diffusion of n1 in every direction (Gradient independent)
                 eps1=0.0,
                 # Self-diffusion of n2 in every direction (Gradient independent)
                 eps2=0.0,
                 # Coordinates of the center of n1
                 center1 = (0.55, 0.55),
                 # Coordinates of the center of n2
                 center2 = (0.45, 0.45),
                 # Peak initial density of n1
                 peak1 = 0.03,
                 # Peak initial density of n2
                 peak2 = 0.03,
                 # Variance of the initial density of n1
                 sigma1 = 0.05,
                 # Variance of the initial density of n2
                 sigma2 = 0.05,
                 # Minimal initial density of n1 on the grid
                 background1 = 0.0,
                 # Minimal initial density of n2 on the grid
                 background2 = 0.0
                 ):
        """
        TissueSim2D

        Class for the 2D-simulation of the dynamic growth of two competing cell populations
        (n1 and n2) based on a continuous model.
        The model takes growth, competition, pressure diffussion and self-diffusion into account.

        Usage:
            sim = TissueSim2D()  # Standard parameters
            sim = TissueSim2D(grid_size=100, G1=1.5, G2=0.8,...)  # Adjusted parameters
            sim.run()  # Run simulation
            sim.visualize()  # Visualize results
            sim.save("name") # Save simulation
            sim.load("name") # Load simulation

        Parameters:
            - Spatial: grid_size (number of divisions of [0,1]^2)
            - Time: count_timesteps (number of time steps), dt (length of a time step), snapscore (storage rate)
            - Physical: beta1, beta2 (pressure sensitivity), eps1, eps2 (self-diffusion), bound_cond (boundary condition)
            - Biological: G1, G2 (growth rate), comp_koef1, comp_koef2 (competition)
            - Initialization: center1, center2, (Coordinates of initial populations) peak1, peak2 (peak initial density), sigma1, sigma2 (variance), background1, background2 (additional initial density on grid)
        """
    

        # Space ===========================================================================>

        # Number of divisions of the grid
        self.grid_size = grid_size
        # Size of a control volume in X-direction
        self.dx = 1.0 / (grid_size - 1)
        # Size of a control volume in Y-direction
        self.dy = self.dx

        # =================================================================================>


        # Time ============================================================================>

        # Number of time steps
        self.count_timesteps = count_timesteps
        # Length of a time step
        self.dt = dt
        # Current time step in the simulation
        self.current_timestep = 0
        # Every [snapscore] time step is saved
        self.snapscore = snapscore

        # =================================================================================>


        # Cell movement ===================================================================>

        # Pressure sensitivity of n1 (Higher beta -> More movement due to pressure)
        self.beta1 = beta1
        # Pressure sensitivity of n2 (Higher beta -> More movement due to pressure)
        self.beta2 = beta2

        # Self-diffusion of n1 in every direction (gradient independent)
        self.eps1 = eps1
        # Self-diffusion of n2 in every direction (gradient independent)
        self.eps2 = eps2

        # Boundary condition ("neumann", "dirichlet")
        self.bound_cond = bound_cond

        # =================================================================================>


        # Growth and competition ==========================================================>

        # Growth rate of n1
        self.G1 = G1
        # Growth rate of n2
        self.G2 = G2

        # Growth inhibition of n1 through n2 (Ability to compete of n1)
        self.comp_koef1 = comp_koef1
        # Growth inhibition of n2 through n1 (Ability to compete of n2)
        self.comp_koef2 = comp_koef2

        # =================================================================================>


        # Densities =======================================================================>

        # Current density (High precision for calculation)
        self.n1 = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.n2 = np.zeros((grid_size, grid_size), dtype=np.float64)

        # Saved density (Saved with lower precision for less memory usage)
        self.n1_snap = np.zeros((grid_size, grid_size, count_timesteps//self.snapscore), dtype=np.float32)
        self.n2_snap = np.zeros((grid_size, grid_size, count_timesteps//self.snapscore), dtype=np.float32)

        # Coordinates of the center of n1
        self.center1 = center1
        # Coordinates of the center of n2
        self.center2 = center2
        # Peak initial density of n1
        self.peak1 = peak1
        # Peak initial density of n2
        self.peak2 = peak2
        # Variance of the initial density of n1
        self.sigma1 = sigma1
        # Variance of the initial density of n2
        self.sigma2 = sigma2
        # Minimal initial density of n1 everywhere
        self.background1 = background1
        # Minimal initial density of n2 everywhere
        self.background2 = background2

        # Initialization of cell densities using the gaussian distribution in 2D
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        gauss1 = self.peak1 * np.exp(-((X - self.center1[1])**2 + (Y - self.center1[0])**2) / (2 * self.sigma1**2))
        gauss2 = self.peak2 * np.exp(-((X - self.center2[1])**2 + (Y - self.center2[0])**2) / (2 * self.sigma2**2))

        # Combination of gaussian and minimal densities
        self.n1[:, :] = gauss1 + self.background1
        self.n2[:, :] = gauss2 + self.background2

        # =================================================================================>
    
    # Compute densities of next time step
    def update_n(self):
        """
        update_n

        Computes the distribution of the densities for the next time step with finite volumes for
        spatial discretization and a explicit euler scheme for temporal discretization.
        Implements pressure diffusion, self-diffusion, growth and competition.

        Usage:
            Is called automatically in run()
            Can be called manually

        Internal variables:
            - n_bar: Reskaled aggregated density
            - v1_x, v1_y, v2_x, v2_y: Velocity fields
            - F1_x, F1_y, F2_x, F2_y: Upwind fluxes
            - div1, div2: Divergence of the fluxes
            - lap1, lap2: Laplace operator for diffusion
            - W1, W2: Growth functions
            - R1, R2: Replication functions
            - n1_new, n2_new: New densities
        """
        # Reskaled aggregated density =====================================================>

        n_bar = (self.n1 + self.n2) / 2

        # =================================================================================>


        # Velocity fields =================================================================>
        # n1 and n2 move along the gradient of the pressure
        # v = -beta * grad(n_bar)

        # n1
        v1_x = np.zeros((self.grid_size+1, self.grid_size))
        v1_x[1:-1, :] = -self.beta1 * (n_bar[1:, :] - n_bar[:-1, :]) / self.dx
        v1_y = np.zeros((self.grid_size, self.grid_size+1))
        v1_y[:, 1:-1] = -self.beta1 * (n_bar[:, 1:] - n_bar[:, :-1]) / self.dy

        # n2
        v2_x = np.zeros((self.grid_size+1, self.grid_size))
        v2_x[1:-1, :] = -self.beta2 * (n_bar[1:, :] - n_bar[:-1, :]) / self.dx
        v2_y = np.zeros((self.grid_size, self.grid_size+1))
        v2_y[:, 1:-1] = -self.beta2 * (n_bar[:, 1:] - n_bar[:, :-1]) / self.dy

        # =================================================================================>


        # Upwind-Flux =====================================================================>
        # Upwind-Scheme for stabilization of the numerical solution
        # F = v * n, direction depending on the sign of v

        # n1
        n1_left = self.n1[:-1, :]
        n1_right = self.n1[1:, :]
        F1_x = np.zeros_like(v1_x)
        F1_x[1:-1, :] = np.where(v1_x[1:-1, :] >= 0, v1_x[1:-1, :] * n1_left, v1_x[1:-1, :] * n1_right)

        n1_down = self.n1[:, :-1]
        n1_up = self.n1[:, 1:]
        F1_y = np.zeros_like(v1_y)
        F1_y[:, 1:-1] = np.where(v1_y[:, 1:-1] >= 0, v1_y[:, 1:-1] * n1_down, v1_y[:, 1:-1] * n1_up)

        # n2
        n2_left = self.n2[:-1, :]
        n2_right = self.n2[1:, :]
        F2_x = np.zeros_like(v2_x)
        F2_x[1:-1, :] = np.where(v2_x[1:-1, :] >= 0, v2_x[1:-1, :] * n2_left, v2_x[1:-1, :] * n2_right)

        n2_down = self.n2[:, :-1]
        n2_up = self.n2[:, 1:]
        F2_y = np.zeros_like(v2_y)
        F2_y[:, 1:-1] = np.where(v2_y[:, 1:-1] >= 0, v2_y[:, 1:-1] * n2_down, v2_y[:, 1:-1] * n2_up)

        # =================================================================================>


        # Divergence ======================================================================>
        # Divergence of flux (finite-volume-discretization)
        # div(F)

        div1 = (F1_x[1:, :] - F1_x[:-1, :]) / self.dx + (F1_y[:, 1:] - F1_y[:, :-1]) / self.dy
        div2 = (F2_x[1:, :] - F2_x[:-1, :]) / self.dx + (F2_y[:, 1:] - F2_y[:, :-1]) / self.dy

        # =================================================================================>


        # Self-diffusion ==================================================================>
        # eps * grad(n)

        lap1 = np.zeros_like(self.n1)
        lap2 = np.zeros_like(self.n2)
        lap1[1:-1, 1:-1] = (
            self.n1[2:, 1:-1] + self.n1[:-2, 1:-1] + self.n1[1:-1, 2:] + self.n1[1:-1, :-2]
            - 4*self.n1[1:-1, 1:-1]
        ) / self.dx**2
        lap2[1:-1, 1:-1] = (
            self.n2[2:, 1:-1] + self.n2[:-2, 1:-1] + self.n2[1:-1, 2:] + self.n2[1:-1, :-2]
            - 4*self.n2[1:-1, 1:-1]
        ) / self.dx**2

        # =================================================================================>

        # Replication with asymetric competition ==========================================>
        # R = n * growth * (1 - n - comp_koef * n_other)

        # Growth functions
        W1 = self.G1 * (1 - self.n1 - self.comp_koef1 * self.n2)
        W2 = self.G2 * (1 - self.n2 - self.comp_koef2 * self.n1)
        # Replication functions
        R1 = self.n1 * W1
        R2 = self.n2 * W2

        # =================================================================================>

        # Update ==========================================================================>
        # Explicit euler scheme:
        # n_new = n_old + dt * ( -div(F) + eps*div(n) + R )

        n1_new = self.n1 + self.dt * (-div1 + self.eps1*lap1 + R1)
        n2_new = self.n2 + self.dt * (-div2 + self.eps2*lap2 + R2)

        # =================================================================================>

        # Boundary conditions =============================================================>

        if self.bound_cond == "neumann":
            n1_new[0, :] = n1_new[1, :]
            n1_new[-1, :] = n1_new[-2, :]
            n1_new[:, 0] = n1_new[:, 1]
            n1_new[:, -1] = n1_new[:, -2]

            n2_new[0, :] = n2_new[1, :]
            n2_new[-1, :] = n2_new[-2, :]
            n2_new[:, 0] = n2_new[:, 1]
            n2_new[:, -1] = n2_new[:, -2]
        elif self.bound_cond == "dirichlet":
            n1_new[0, :] = n1_new[-1, :] = n1_new[:, 0] = n1_new[:, -1] = 0
            n2_new[0, :] = n2_new[-1, :] = n2_new[:, 0] = n2_new[:, -1] = 0
        
        # =================================================================================>

        # No negative values ==============================================================>
        np.maximum(n1_new, 0.0, out=n1_new)
        np.maximum(n2_new, 0.0, out=n2_new)

        # =================================================================================>


        # Overwrite =======================================================================>

        self.n1 = n1_new
        self.n2 = n2_new

        # =================================================================================>

    # Run simulation
    def run(self):
        """
        run

        Runs the simulation over all time steps. Saves the state of the densities 
        of the cells in regular time intervals and prints the progress.

        Usage:
            sim.run()  # Starts simulation

        Variables:
            - t: Current time step
        """
        # For every time step
        for t in range(self.count_timesteps):
            # Save current time step
            self.current_timestep = t
            # Compute new densities
            self.update_n()
            # Save the state of densities and show the progress in regular time intervals.
            if (t%self.snapscore == 0):
                self.n1_snap[:, :, self.current_timestep//self.snapscore] = self.n1
                self.n2_snap[:, :, self.current_timestep//self.snapscore] = self.n2
                #print(f"{t/self.count_timesteps*100:.2f}% - total_mass={self.n1.sum() + self.n2.sum():.6e}") # Print total mass
                print(f"{t/self.count_timesteps*100:.2f}%")
        print(f"100.00%")
        print(f"Simulation completed. Total_mass={self.n1.sum() + self.n2.sum():.6e}")
    
    # Plot outcome of the simulation with interactive sliders
    def visualize(self):
        """
        Interactive visualization with linear color coding, contours, sliders and legend.
        """
        # Data
        timesteps = self.n1_snap.shape[2]

        n1_max = self.n1_snap.max()
        n2_max = self.n2_snap.max()
        n12_max = (self.n1_snap + self.n2_snap).max()
        max_val = max(n1_max, n2_max)

        n1 = self.n1_snap[:, :, 0]
        n2 = self.n2_snap[:, :, 0]
        n12 = n1 + n2

        # Linear color maps
        red_linear = LinearSegmentedColormap.from_list(
            "red_linear", [(1, 1, 1), (1, 0, 0)]
        )
        blue_linear = LinearSegmentedColormap.from_list(
            "blue_linear", [(1, 1, 1), (0, 0, 1)]
        )
        purple_linear = LinearSegmentedColormap.from_list(
            "purple_linear", [(1, 1, 1), (0.5, 0, 0.5)]
        )

        # Figure
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        plt.subplots_adjust(bottom=0.40, right=0.85)

        # Densities
        im0 = axes[0].imshow(n1, origin="lower", extent=[0, 1, 0, 1], cmap=red_linear, vmin=0, vmax=n1_max)
        axes[0].set_title("n1")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(n2, origin="lower", extent=[0, 1, 0, 1], cmap=blue_linear, vmin=0, vmax=n2_max)
        axes[1].set_title("n2")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(n12, origin="lower", extent=[0, 1, 0, 1], cmap=purple_linear, vmin=0, vmax=n12_max)
        axes[2].set_title("n1 + n2")
        fig.colorbar(im2, ax=axes[2])

        # RGB-Mix
        mixed_rgb = np.ones((self.grid_size, self.grid_size, 3))
        n1_norm = np.clip(n1 / max_val, 0, 1)
        n2_norm = np.clip(n2 / max_val, 0, 1)
        mixed_rgb[..., 1] -= n1_norm
        mixed_rgb[..., 2] -= n1_norm
        mixed_rgb[..., 0] -= n2_norm
        mixed_rgb[..., 1] -= n2_norm
        mixed_rgb = np.clip(mixed_rgb, 0, 1)

        im3 = axes[3].imshow(mixed_rgb, origin="lower", extent=[0, 1, 0, 1])
        axes[3].set_title("n1 (rot) / n2 (blau)")

        # Parameter legend
        param_text = (
            f"grid_size: {self.grid_size}\n"
            f"bound_cond: {self.bound_cond}\n"
            f"G1: {self.G1:.2f}, G2: {self.G2:.2f}\n"
            f"comp_koef1: {self.comp_koef1:.2f}, comp_koef2: {self.comp_koef2:.2f}\n"
            f"beta1: {self.beta1:.2f}, beta2: {self.beta2:.2f}\n"
            f"eps1: {self.eps1:.2f}, eps2: {self.eps2:.2f}"
        )
        axes[4].axis("off")
        axes[4].text(0.05, 0.5, param_text, fontsize=12, verticalalignment='center', 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        axes[4].set_title("Parameter")

        # Slider & Controls
        ax_slider_t = plt.axes([0.25, 0.22, 0.5, 0.03])
        slider_t = Slider(ax_slider_t, "Time step", 0, timesteps - 1, valinit=0, valstep=1)

        ax_slider_p = plt.axes([0.25, 0.17, 0.5, 0.03])
        slider_p = Slider(ax_slider_p, "Mass [%]", 50, 99, valinit=85)

        ax_check = plt.axes([0.05, 0.12, 0.12, 0.06])
        check = CheckButtons(ax_check, ["Contours"], [True])

        show_contours = True
        contours_n1, contours_n2, contours_n12 = [], [], []

        def mass_threshold(data, fraction):
            flat = data.ravel()
            idx = np.argsort(flat)[::-1]
            cumsum = np.cumsum(flat[idx])
            return flat[idx][np.searchsorted(cumsum, fraction * cumsum[-1])]

        def clear_contours():
            for l in contours_n1 + contours_n2 + contours_n12:
                l.remove()
            contours_n1.clear()
            contours_n2.clear()
            contours_n12.clear()

        def update(val=None):
            t = int(slider_t.val)
            frac = slider_p.val / 100.0

            n1 = self.n1_snap[:, :, t]
            n2 = self.n2_snap[:, :, t]
            n12 = n1 + n2

            im0.set_data(n1)
            im1.set_data(n2)
            im2.set_data(n12)

            n1_norm = np.clip(n1 / max_val, 0, 1)
            n2_norm = np.clip(n2 / max_val, 0, 1)

            mixed_rgb[:] = 1.0
            mixed_rgb[..., 1] -= n1_norm
            mixed_rgb[..., 2] -= n1_norm
            mixed_rgb[..., 0] -= n2_norm
            mixed_rgb[..., 1] -= n2_norm
            mixed_rgb[:] = np.clip(mixed_rgb, 0, 1)
            im3.set_data(mixed_rgb)

            clear_contours()
            if show_contours:
                thr1 = mass_threshold(n1, frac)
                thr2 = mass_threshold(n2, frac)
                thr12 = mass_threshold(n12, frac)

                for c in find_contours(n1, thr1):
                    c /= (self.grid_size - 1)
                    contours_n1.append(axes[0].plot(c[:, 1], c[:, 0], color=(0.5,0,0), lw=1.5)[0])
                    contours_n1.append(axes[3].plot(c[:, 1], c[:, 0], color=(0.5,0,0), lw=1.5)[0])

                for c in find_contours(n2, thr2):
                    c /= (self.grid_size - 1)
                    contours_n2.append(axes[1].plot(c[:, 1], c[:, 0], color=(0,0,0.5), lw=1.5)[0])
                    contours_n2.append(axes[3].plot(c[:, 1], c[:, 0], color=(0,0,0.5), lw=1.5)[0])

                for c in find_contours(n12, thr12):
                    c /= (self.grid_size - 1)
                    contours_n12.append(axes[2].plot(c[:, 1], c[:, 0], "k", lw=1.5)[0])

            fig.canvas.draw_idle()

        def toggle(label):
            nonlocal show_contours
            show_contours = not show_contours
            update()

        slider_t.on_changed(update)
        slider_p.on_changed(update)
        check.on_clicked(toggle)

        update()
        plt.show()

    # Save simulation
    def save(self, filename=""):
        """
        save

        Saves the complete state of the simulation object using pickle.

        Usage:
            sim.save()  # Asks for filename
            sim.save("my_sim.pkl")  # Saves directly with specified filename

        Variables:
            - filename: Path and filename
        """
        # If no filename was hand over ask in terminal
        if filename == "":
            filename = input("filename to save:")

        # Save object with pickle
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Load simulation
    def load(self, filename=""):
        """
        load

        Loads a previously saved simulation and overwrites the whole state of the object.
        Verifies the existence of the specified file and type.

        Usage:
            sim.load()  # Asks for filename
            sim.load("meine_simulation.pkl")  # Loads directly with specified filename

        Variablen:
            - filename: Path and filename
        """
        # If no filename was hand over ask in terminal
        if filename == "":
            filename = input("filename to load:")

        # Verify existence of file
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")

        # Load object from the file
        with open(filename, "rb") as f:
            loaded = pickle.load(f)

        # Verify type
        if not isinstance(loaded, TissueSim2D):
            raise TypeError("Loaded file does not contain TissueSim2D-object.")

        # Replace internal state with loaded state
        self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)

if __name__ == "__main__":
    # Create simulation object with standard parameters
    sim = TissueSim2D()

    # Run simulation
    sim.run()

    # Save simulation
    sim.save()

    # Load simulation
    #sim.load()

    # Visualize simulation
    sim.visualize()
