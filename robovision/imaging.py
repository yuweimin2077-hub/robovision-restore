"""Known-kernel motion blur and two inverse-problem solvers.

Images are RGB floats. All operators use periodic boundary conditions.
The sensor simulation deliberately does NOT clip Gaussian noise: clipping
would change y = Hx + n. Outputs are clipped only for display and evaluation.
"""

import numpy as np


def motion_psf(length=13, angle=25.0):
    """A normalized, bilinearly sampled straight-line motion kernel."""
    if length < 1 or int(length) != length:
        raise ValueError("length must be a positive integer")
    length = int(length)
    size = length if length % 2 else length + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    theta = np.deg2rad(angle)
    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, max(1, length * 8)):
        x, y = center + t * np.cos(theta), center + t * np.sin(theta)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        for yy in (y0, y0 + 1):
            for xx in (x0, x0 + 1):
                if 0 <= xx < size and 0 <= yy < size:
                    kernel[yy, xx] += max(0, 1 - abs(x - xx)) * max(0, 1 - abs(y - yy))
    return kernel / kernel.sum()


def psf_to_otf(kernel, shape):
    """Center the PSF at Fourier origin; avoid the common one-pixel shift bug."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or any(k > s for k, s in zip(kernel.shape, shape)):
        raise ValueError("kernel must be 2D and fit inside the image")
    padded = np.zeros(shape, dtype=np.float64)
    padded[:kernel.shape[0], :kernel.shape[1]] = kernel
    for axis, size in enumerate(kernel.shape):
        padded = np.roll(padded, -(size // 2), axis=axis)
    return np.fft.fft2(padded)


def blur(image, kernel):
    otf = psf_to_otf(kernel, image.shape[:2])
    if image.ndim == 3:
        otf = otf[..., None]
    return np.fft.ifft2(np.fft.fft2(image, axes=(0, 1)) * otf, axes=(0, 1)).real


def degrade(image, kernel, sigma=0.01, seed=0):
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    return blur(image, kernel) + np.random.default_rng(seed).normal(0, sigma, image.shape)


def wiener_restore(observed, kernel, balance=0.003):
    """Wiener-form/Tikhonov filter: conj(H)Y / (|H|^2 + balance).

    balance is a development-set-tuned regularization constant, not an
    estimated noise-to-signal power ratio. No clean test image is used.
    """
    if balance <= 0:
        raise ValueError("balance must be positive")
    h = psf_to_otf(kernel, observed.shape[:2])
    filt = h.conj() / (np.abs(h) ** 2 + balance)
    if observed.ndim == 3:
        filt = filt[..., None]
    return np.fft.ifft2(np.fft.fft2(observed, axes=(0, 1)) * filt, axes=(0, 1)).real


def gradient(image):
    return np.stack((np.roll(image, -1, axis=0) - image,
                     np.roll(image, -1, axis=1) - image), axis=-1)


def gradient_adjoint(field):
    return (np.roll(field[..., 0], 1, axis=0) - field[..., 0]
            + np.roll(field[..., 1], 1, axis=1) - field[..., 1])


def tv_admm(observed, kernel, lam=0.003, rho=0.15, iterations=80):
    """Minimize 0.5 ||Hx-y||^2 + lam * sum_{pixel,channel} ||Dx||_2.

    Split z=Dx. The x step is solved by FFT, z by vector soft thresholding,
    and u is the scaled dual variable. RGB channels are regularized separately.
    The unconstrained iterate is retained; clipping is a separate display step.
    Returns (restored, history), including objective and constraint residuals.
    """
    if lam < 0 or rho <= 0 or iterations < 1:
        raise ValueError("lam >= 0, rho > 0, iterations >= 1 are required")
    y = np.asarray(observed, dtype=np.float64)
    height, width = y.shape[:2]
    h = psf_to_otf(kernel, (height, width))
    lap = (4 * np.sin(np.pi * np.fft.fftfreq(height))[:, None] ** 2
           + 4 * np.sin(np.pi * np.fft.fftfreq(width))[None, :] ** 2)
    denominator = np.abs(h) ** 2 + rho * lap
    if y.ndim == 3:
        h, denominator = h[..., None], denominator[..., None]
    data_rhs = h.conj() * np.fft.fft2(y, axes=(0, 1))
    x = y.copy()
    z = gradient(x)
    u = np.zeros_like(z)
    history = []
    for step in range(iterations):
        rhs = data_rhs + rho * np.fft.fft2(gradient_adjoint(z - u), axes=(0, 1))
        x = np.fft.ifft2(rhs / np.maximum(denominator, 1e-12), axes=(0, 1)).real
        dx = gradient(x)
        v = dx + u
        norm = np.sqrt(np.sum(v * v, axis=-1, keepdims=True))
        z_old = z
        z = np.maximum(0, 1 - lam / (rho * np.maximum(norm, 1e-12))) * v
        u = u + dx - z
        if step % 5 == 0 or step == iterations - 1:
            residual = blur(x, kernel) - y
            objective = 0.5 * np.sum(residual ** 2) + lam * np.sum(np.linalg.norm(dx, axis=-1))
            history.append({"iteration": step + 1, "objective": float(objective),
                            "primal_rms": float(np.sqrt(np.mean((dx - z) ** 2))),
                            "dual_rms": float(rho * np.sqrt(np.mean(gradient_adjoint(z - z_old) ** 2)))})
    return x, history
