import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from sklearn.metrics.pairwise import pairwise_kernels


def kernel_smoothing(
    X, u,
    kernel_metric="rbf",
    kernel_params={"gamma":1},
    _lambda=1e-5,
):
    """
    X = (J, N) array containing mesh observation points (i.e. assumes N-D grid)
    u = (J,) array of PDE solutions
    kernel_metric = kernel function to use
    kernel_params = dictionary of kernel parameters
    _lambda = assumed standard deviation of measurement noise
    """
    K = lambda x, y: pairwise_kernels(x, Y=y, metric=kernel_metric, **kernel_params)
    z = np.linalg.lstsq(K(X, X) + (_lambda**2) * np.eye(len(X)), u, rcond=None)[0]
    u_bar = lambda x: K(x, X).dot(z)
    return u_bar


def kernel_diff(
    X, u, d,
    kernel_metric="rbf",
    kernel_params={"gamma":1},
    _lambda=1e-5,
):
    """
    X = (J, N) array containing mesh observation points (i.e. assumes N-D grid)
    u = (J,) array of PDE solutions
    d = order of derivative
    Note: Only RBF kernel functionality is implemented.
    """
    Kxx = pairwise_kernels(X, Y=X, metric=kernel_metric, **kernel_params)
    z = np.linalg.lstsq(Kxx + (_lambda**2) * np.eye(len(X)), u, rcond=None)[0]
    if kernel_metric == "rbf": # K(x, y) = exp(-gamma||x-y||^2)
        u_bar = lambda x: jnp.array(z).dot(
            jnp.exp(-kernel_params["gamma"] * jnp.sum((x-X)**2, axis=1).reshape(-1, 1))
        )[0]
    else:
        raise ValueError("Kernel method not supported for differentiation.")
    # Compute derivatives.
    du_bar = grad(u_bar)
    for _ in range(d - 1):
        du_bar = grad(du_bar)
    return du_bar


def kernel_regression(
    S, f,
    K=None,
    kernel_metric="poly",
    kernel_params={"gamma":1,"degree":3,"coef0":1},
    _lambda=1e-5,
):
    """
    S = (J, Jp) array of state variables
    f = (J,) array of corresponding PDE forcing terms
    K = custom kernel function, if given
    """
    if K is None:
        K = lambda x, y: pairwise_kernels(x, Y=y, metric=kernel_metric, **kernel_params)
    z = np.linalg.lstsq(K(S, S) + (_lambda**2) * np.eye(len(S)), f, rcond=None)[0]
    P = lambda s: K(s, S).dot(z)
    return P


def test_step1(a=4, noise_mag=0.0):
    """
    Benchmarking for kernel smoothing and differentiation step.
    Benchmarks against the function u(x) = sin(ax).
    """
    # Evaluate true function values.
    x = np.linspace(0, 4 * np.pi, 100)
    u_true = np.sin(a * x)
    ux_true = a * np.cos(a * x)
    uxx_true = -(a ** 2) * np.sin(a * x)
    uxxx_true = -(a ** 3) * np.cos(a * x)

    # Add noise to data?
    u_noisy = u_true + (noise_mag * np.random.randn(*u_true.shape))

    # Get functions.
    u_func = kernel_smoothing(x[:, None], u_noisy)
    ux_func = kernel_diff(x[:, None], u_noisy, d=1)
    uxx_func = kernel_diff(x[:, None], u_noisy, d=2)
    uxxx_func = kernel_diff(x[:, None], u_noisy, d=3)

    # Evaluate function values.
    u_est = u_func(x[:, None])
    ux_est = np.array([ux_func(np.array(xi)) for xi in x])
    uxx_est = np.array([uxx_func(np.array(xi)) for xi in x])
    uxxx_est = np.array([uxxx_func(np.array(xi)) for xi in x])

    # Plot the computed derivatives.
    plt.figure(figsize=(14, 2))
    for i, (truth, est, label) in enumerate(
        zip([u_true, ux_true, uxx_true, uxxx_true],
            [u_est, ux_est, uxx_est, uxxx_est],
            ["u", "ux", "uxx", "uxxx"])
    ):
        plt.subplot(1, 4, i + 1)
        plt.plot(x, truth, c="k", label="Truth")
        plt.plot(x, est, "--", c="r", label="Approx")
        plt.xlabel("x")
        plt.ylabel(label)
        plt.legend()
    plt.tight_layout()
    plt.show()


def test_step2(A=1, B=0, k=10):
    """
    Benchmarking for kernel regression step.
    Benchmarks against u(t) = Acos(kt) + Bsin(kt), which satisfies u_tt + (k^2)u = 0.
    """
    # Define solution to the wave equation (and respective derivatives).
    t = np.linspace(0, 4 * np.pi, 100)
    U_wave = (A * np.cos(k * t)) + (B * np.sin(k * t))
    Ut_wave = -(k * A * np.sin(k * t)) + (k * B * np.cos(k * t))
    Utt_wave = -((k ** 2) * A * np.cos(k * t)) - ((k ** 2) * B * np.sin(k * t))

    # Define forcing and S vectors.
    F_wave = -(k ** 2) * U_wave
    S_wave = np.vstack([t, U_wave, Ut_wave, Utt_wave]).T

    # Regress PDE.
    P_wave = kernel_regression(
        S_wave,
        F_wave,
        kernel_metric="poly",
        kernel_params={"gamma":1, "degree":3, "coef0":0.015},
        _lambda=1e-5,
    )

    # Evaluate at collocation points.
    P_approx = P_wave(S_wave)
    P_true = Utt_wave

    # Plot and evaluate results.
    plt.figure(figsize=(6, 2))
    plt.title("P(S)")
    plt.plot(t, P_true, c="k", label="Truth")
    plt.plot(t, P_approx, "--", c="r", label="Approx")
    plt.legend()
    plt.show()
