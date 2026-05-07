"""
uncertainty_ellipsoid.py
========================
Calcula a elipsoide de incerteza 3D de cada ponto triangulado
via propagação de covariância pelo Jacobiano numérico da triangulação.

Contexto do projeto: o tracker é o AprilTag. Para cada tag detectada,
os pontos rastreados são:
    - tag{id}_center     → centro geométrico da tag
    - tag{id}_corner0..3 → 4 cantos da tag

Para CADA um desses pontos, esta função estima uma elipsoide 3D de
incerteza com base na propagação do erro de detecção em pixels.

Retorna:
    ellipsoids_frame_marker: lista de dicts com:
        - center: np.ndarray [3]    — coordenadas 3D do ponto
        - radii:  np.ndarray [3]    — semi-eixos 1-sigma (em mm, menor → maior)
        - evecs:  np.ndarray [3,3]  — autovetores (orientação da elipsoide)
        - Sigma:  np.ndarray [3,3]  — matriz de covariância 3x3 completa
"""

import logging
import multiprocessing
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _triangulate_flat(cgroup, p2d_flat: np.ndarray) -> np.ndarray:
    """
    Triangula UM ponto dado um array p2d_flat de shape [n_cams*2].
    Retorna array shape [3] com as coordenadas 3D.
    """
    n_cams = p2d_flat.shape[0] // 2
    pts = p2d_flat.reshape(n_cams, 1, 2)  # shape esperado por triangulate
    result = cgroup.triangulate(pts, progress=False)  # shape [1, 3]
    return result[0]


def compute_jacobian_numerical(
    cgroup,
    p2d_per_cam: np.ndarray,
    eps: float = 0.5,
) -> np.ndarray:
    """
    Calcula o Jacobiano numérico J = ∂(3D point) / ∂(pixels) por diferenças finitas.

    Args:
        cgroup:        objeto CameraGroup do anipose
        p2d_per_cam:  shape [n_cams, 2] — coordenadas 2D do ponto em cada câmera
        eps:           perturbação em pixels (padrão: 0.5 px)

    Returns:
        J: shape [3, 2*n_cams]  — Jacobiano
    """
    n_cams = p2d_per_cam.shape[0]
    n_params = 2 * n_cams
    x0 = p2d_per_cam.ravel().astype(float)

    J = np.zeros((3, n_params))
    for j in range(n_params):
        x_p = x0.copy(); x_p[j] += eps
        x_m = x0.copy(); x_m[j] -= eps
        p3d_p = _triangulate_flat(cgroup, x_p)
        p3d_m = _triangulate_flat(cgroup, x_m)
        if np.any(np.isnan(p3d_p)) or np.any(np.isnan(p3d_m)):
            J[:, j] = np.nan
        else:
            J[:, j] = (p3d_p - p3d_m) / (2.0 * eps)

    return J


def compute_covariance_3d(J: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """
    Propaga a incerteza dos pixels para o espaço 3D.

    Σ_P = J · (σ² · I) · Jᵀ = σ² · J · Jᵀ

    Args:
        J:             Jacobiano shape [3, 2*n_cams]
        sigma_pixels:  desvio padrão assumido do detector em pixels

    Returns:
        Sigma_P: covariância 3x3 em mm² (ou unidades do espaço 3D)
    """
    valid_cols = ~np.any(np.isnan(J), axis=0)
    J_clean = J[:, valid_cols]
    if J_clean.shape[1] == 0:
        return np.full((3, 3), np.nan)
    return (sigma_pixels ** 2) * (J_clean @ J_clean.T)


def ellipsoid_from_covariance(Sigma: np.ndarray):
    """
    Extrai os semi-eixos e a orientação de uma covariância 3x3.

    Returns:
        radii:  semi-eixos (1-sigma) em mm — shape [3], ordenados crescente
        evecs:  autovetores correspondentes — shape [3, 3]
    """
    if np.any(np.isnan(Sigma)):
        return np.full(3, np.nan), np.eye(3)
    eigvals, evecs = np.linalg.eigh(Sigma)  # eigh para matrizes simétricas
    eigvals = np.clip(eigvals, 0, None)      # garante não-negativo
    radii = np.sqrt(eigvals)                 # raios em mm (1-sigma)
    return radii, evecs


def compute_uncertainty_for_all_frames(
    cgroup,
    image_2d_data: np.ndarray,
    sigma_pixels: float = 3.0,
    subsample_frames: int = 1,
    kill_event: Optional[multiprocessing.Event] = None,
) -> list:
    """
    Calcula as elipsoídes de incerteza para todos os frames e pontos.

    Args:
        cgroup:           CameraGroup do anipose (já calibrado)
        image_2d_data:    shape [n_cams, n_frames, n_points, 2]
        sigma_pixels:     desvio padrão assumido do detector em pixels
        subsample_frames: calcula 1 a cada N frames (1 = todos)
        kill_event:       evento para cancelar o cálculo

    Returns:
        lista de comprimento n_frames, onde cada elemento é:
            lista de comprimento n_points, onde cada elemento é dict:
            {
                'center': np.ndarray[3],
                'radii':  np.ndarray[3],   # semi-eixos 1-sigma em mm
                'evecs':  np.ndarray[3,3], # orientação
                'Sigma':  np.ndarray[3,3], # covariância completa
            }
            (ou None se o ponto não tem dados suficientes)
    """
    n_cams, n_frames, n_points, _ = image_2d_data.shape
    results = []

    logger.info(
        f"Calculando elipsoídes de incerteza: "
        f"{n_frames} frames × {n_points} pontos × {n_cams} câmeras"
    )

    for frame_idx in range(n_frames):
        if kill_event is not None and kill_event.is_set():
            logger.info("Cálculo de incerteza cancelado.")
            break

        frame_results = []

        # Pula frames se subsample_frames > 1
        if frame_idx % subsample_frames != 0 and frame_idx != 0:
            results.append(None)  # placeholder para manter indexação por frame
            continue

        # Contadores de diagnóstico para o frame 0
        diag = frame_idx == 0
        n_few_cams, n_nan_center, n_exc, n_ok = 0, 0, 0, 0

        for pt_idx in range(n_points):
            # p2d: shape [n_cams, 2]
            p2d = image_2d_data[:, frame_idx, pt_idx, :]

            # Verifica se há câmeras suficientes com detecção válida
            valid_cams = ~np.isnan(p2d[:, 0])
            if np.sum(valid_cams) < 2:
                n_few_cams += 1
                frame_results.append(None)
                continue

            # Ponto 3D central (da triangulação já feita)
            pts_for_tri = p2d.copy()
            pts_for_tri[~valid_cams] = np.nan

            center = _triangulate_flat(cgroup, pts_for_tri.ravel())
            if np.any(np.isnan(center)):
                n_nan_center += 1
                if diag and n_nan_center == 1:
                    logger.warning(
                        f"[DIAG frame 0 pt {pt_idx}] triangulação retornou NaN. "
                        f"p2d={p2d}  valid_cams={valid_cams}"
                    )
                frame_results.append(None)
                continue

            # Jacobiano e covariância
            try:
                J = compute_jacobian_numerical(cgroup, pts_for_tri, eps=0.5)
                Sigma = compute_covariance_3d(J, sigma_pixels)
                radii, evecs = ellipsoid_from_covariance(Sigma)
                n_ok += 1
            except Exception as e:
                n_exc += 1
                logger.warning(f"Erro no ponto {pt_idx} frame {frame_idx}: {e}")
                frame_results.append(None)
                continue

            frame_results.append({
                "center": center,
                "radii": radii,
                "evecs": evecs,
                "Sigma": Sigma,
            })

        if diag:
            logger.info(
                f"[DIAG frame 0] poucos_cams={n_few_cams}  nan_center={n_nan_center}  "
                f"excecoes={n_exc}  ok={n_ok}  "
                f"2D data shape={image_2d_data.shape}  "
                f"amostra p2d[pt0]={image_2d_data[:, 0, 0, :]}"
            )

        results.append(frame_results)

    logger.info("Cálculo de elipsoídes concluído.")

    # Resumo diagnóstico: quantos frames têm pelo menos 1 elipsoíde válida
    n_frames_with_data = sum(
        1 for fr in results
        if fr is not None and any(pt is not None for pt in fr)
    )
    logger.info(
        f"[DIAG resumo] {n_frames_with_data}/{len(results)} frames têm pelo menos 1 elipsoíde válida"
    )

    return results
