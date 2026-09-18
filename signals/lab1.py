import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import IntegrationWarning, quad


warnings.simplefilter("ignore", IntegrationWarning)




class FourierApproximation:


    def __init__(self, n_variant, interval, n_terms=10):
        self.n = n_variant
        self.a, self.b = interval
        self.L = (self.b - self.a) / 2.0
        self.N = n_terms
        self.a_k = [0.0] * (self.N + 1)
        self.b_k = [0.0] * (self.N + 1)
        self.compute_coefficients()


    def f(self, x):
        if self.n % 2 == 0:
            return (x**self.n) * math.exp(-(x**2) / self.n)
        else:
            return self.n * math.sin(math.pi * self.n * x)


    def _f_vec(self, x_arr):
        return np.array([self.f(x) for x in x_arr])


    def compute_coefficients(self):
        a0, _ = quad(lambda x: self.f(x), self.a, self.b, limit=200)
        self.a_k[0] = a0 / self.L
        self.b_k[0] = 0.0


        for k in range(1, self.N + 1):
            ak, _ = quad(
                lambda x: self.f(x)
                * math.cos(k * math.pi * (x - self.a) / self.L - math.pi),
                self.a,
                self.b,
                limit=200,
            )
            bk, _ = quad(
                lambda x: self.f(x)
                * math.sin(k * math.pi * (x - self.a) / self.L - math.pi),
                self.a,
                self.b,
                limit=200,
            )
            self.a_k[k] = ak / self.L
            self.b_k[k] = bk / self.L


    def evaluate_fourier(self, x, N_order=None):
        if N_order is None or N_order > self.N:
            N_order = self.N


        val = self.a_k[0] / 2.0
        for k in range(1, N_order + 1):
            arg = k * math.pi * (x - self.a) / self.L - math.pi
            val += self.a_k[k] * math.cos(arg) + self.b_k[k] * math.sin(arg)
        return val


    def evaluate_fourier_vec(self, x_arr, N_order=None):
        return np.array([self.evaluate_fourier(x, N_order) for x in x_arr])


    def calculate_relative_error(self, num_points=1000):
        x_vals = np.linspace(self.a, self.b, num_points)
        y_exact = self._f_vec(x_vals)
        y_approx = self.evaluate_fourier_vec(x_vals)


        norm_diff = np.sqrt(np.trapezoid((y_exact - y_approx) ** 2, x_vals))
        norm_exact = np.sqrt(np.trapezoid(y_exact**2, x_vals))


        if norm_exact == 0:
            return 0.0
        return norm_diff / norm_exact


    def plot_harmonics(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))


        k_vals = np.arange(0, self.N + 1)
        axs[0].stem(k_vals, self.a_k, linefmt="b-", markerfmt="bo", basefmt="r-")
        axs[0].set_title("Fourier Cosine Coefficients (a_k)")
        axs[0].set_xlabel("Harmonic index (k)")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True)


        axs[1].stem(k_vals, self.b_k, linefmt="g-", markerfmt="go", basefmt="r-")
        axs[1].set_title("Fourier Sine Coefficients (b_k)")
        axs[1].set_xlabel("Harmonic index (k)")
        axs[1].set_ylabel("Amplitude")
        axs[1].grid(True)


        plt.tight_layout()
        plt.show()


    def plot_approximation(self, num_points=1000):
        x_vals = np.linspace(self.a, self.b, num_points)
        y_exact = self._f_vec(x_vals)
        y_approx = self.evaluate_fourier_vec(x_vals)


        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, y_exact, "b-", label="Exact f(x)", linewidth=2)
        plt.plot(
            x_vals,
            y_approx,
            "r--",
            label=f"Fourier Approx (N={self.N})",
            linewidth=1.5,
        )
        plt.title("Function Approximation via Fourier Series")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()


    def save_results_to_file(self, filename="fourier_results.txt"):
        rel_err = self.calculate_relative_error()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Order N: {self.N}\n")
            f.write(f"Relative Error: {rel_err:.6e}\n\n")
            f.write("Fourier Coefficients:\n")
            f.write(f"a_0 = {self.a_k[0]:.6f}\n")
            for k in range(1, self.N + 1):
                f.write(
                    f"a_{k} = {self.a_k[k]:.6f},  b_{k} = {self.b_k[k]:.6f}\n"
                )
        print(f"Results saved to '{filename}'")




def main():
    student_number = int(input("Enter student journal number (n): "))
    order_input = input("Enter approximation order (N, default 10): ").strip()
    order_N = int(order_input) if order_input else 10


    if student_number % 2 == 0:
        interval = (-math.pi, math.pi)
    else:
        interval = (0, math.pi)


    app = FourierApproximation(student_number, interval, order_N)
    err = app.calculate_relative_error()


    print("\n" + "=" * 40)
    print(f"Approximation Order (N): {app.N}")
    print(f"Relative Error: {err:.6e} ({err * 100:.2f}%)")
    print("=" * 40)


    app.save_results_to_file("fourier_results.txt")
    app.plot_harmonics()
    app.plot_approximation()




if __name__ == "__main__":
    main()
