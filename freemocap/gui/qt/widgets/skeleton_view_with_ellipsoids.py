"""
skeleton_view_with_ellipsoids.py
=================================
Subclasse de SkeletonViewWidget que adiciona a visualização das
elipsoídes de incerteza 3D no gráfico Matplotlib 3D.

Contexto do projeto: o tracker usado é o AprilTag, que detecta
para cada tag os pontos:
    - tag{id}_center     (centro da tag)
    - tag{id}_corner0..3 (4 cantos da tag)

As elipsoídes são calculadas pela propagação de covariância do Jacobiano
numérico da triangulação (ver uncertainty_ellipsoid.py). Cada elipsoíde
representa a incerteza 3D de UM ponto (centro ou canto de uma tag)
com base no erro de detecção em pixels de cada câmera.

Uso no freemocap_main_window.py:
    O widget é injetado automaticamente — veja _create_central_tab_widget.
"""

import logging
from typing import Optional, List

import numpy as np
from skelly_viewer.gui.qt.widgets.skeleton_view_widget import SkeletonViewWidget

logger = logging.getLogger(__name__)


def _make_ellipsoid_surface(center: np.ndarray,
                            radii: np.ndarray,
                            evecs: np.ndarray,
                            n: int = 12):
    """
    Gera os vértices de uma elipsoíde no espaço 3D.

    Args:
        center: shape [3]   — centro
        radii:  shape [3]   — semi-eixos (1-sigma)
        evecs:  shape [3,3] — autovetores (orientação)
        n:      resolução da malha (menor = mais rápido)

    Returns:
        x, y, z: arrays [n, n] dos vértices na superfície da elipsoíde
    """
    # Esfera unitária parametrizada
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    # Aplica escala pelos semi-eixos
    ellipsoid = np.stack([
        radii[0] * sphere_x,
        radii[1] * sphere_y,
        radii[2] * sphere_z,
    ], axis=-1)  # shape [n, n, 3]

    # Rotaciona pela orientação dos autovetores
    rotated = ellipsoid @ evecs.T  # shape [n, n, 3]

    # Translada para o centro
    x = rotated[:, :, 0] + center[0]
    y = rotated[:, :, 1] + center[1]
    z = rotated[:, :, 2] + center[2]

    return x, y, z


class SkeletonViewWithEllipsoids(SkeletonViewWidget):
    """
    Extensão do SkeletonViewWidget com suporte a elipsoídes de incerteza 3D
    para pontos de AprilTag.

    Para cada frame, cada ponto rastreado (centro ou canto de uma tag)
    tem uma elipsoíde associada representando sua incerteza de posição 3D.

    Fluxo:
        1. Os dados 3D são carregados normalmente pelo viewer.
        2. Chame set_uncertainty_ellipsoids(ellipsoids_per_frame) com
           o resultado de compute_uncertainty_for_all_frames().
        3. Ao navegar pelos frames, as elipsoídes são desenhadas junto
           com os pontos das tags.
    """

    def __init__(self):
        super().__init__()
        # Lista: um elemento por frame, cada elemento é lista de dicts
        # (ou None) por ponto, com chaves: center, radii, evecs
        self._ellipsoids_per_frame: Optional[List] = None
        self._ellipsoid_surfaces = []   # handles matplotlib atuais
        self._show_ellipsoids = True
        self._ellipsoid_alpha = 0.25
        self._ellipsoid_color = "cyan"

    # ──────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────

    def set_uncertainty_ellipsoids(self, ellipsoids_per_frame: list):
        """
        Define as elipsoídes pré-calculadas para todos os frames.

        Args:
            ellipsoids_per_frame: saída de compute_uncertainty_for_all_frames()
                Lista de n_frames elementos; cada elemento é uma lista de
                n_points dicts com chaves: center, radii, evecs (ou None)
        """
        self._ellipsoids_per_frame = ellipsoids_per_frame
        logger.info(
            f"Elipsoídes de incerteza carregadas: "
            f"{len(ellipsoids_per_frame)} frames"
        )

    def toggle_ellipsoids(self, visible: bool):
        """Liga/desliga a visualização das elipsoídes."""
        self._show_ellipsoids = visible
        for surf in self._ellipsoid_surfaces:
            if surf is not None:
                surf.set_visible(visible)
        self._figure_widget.figure.canvas.draw_idle()

    def set_ellipsoid_style(self, alpha: float = 0.25, color: str = "cyan"):
        """Ajusta a transparência e cor das elipsoídes."""
        self._ellipsoid_alpha = alpha
        self._ellipsoid_color = color

    # ──────────────────────────────────────────────────────────────────
    # Override do método de atualização do plot
    # ──────────────────────────────────────────────────────────────────

    def update_skeleton_plot(self, frame_number: int):
        """Override: atualiza esqueleto + elipsoídes para o frame."""
        super().update_skeleton_plot(frame_number)  # desenha esqueleto normalmente
        self._update_ellipsoids(frame_number)

    def _initialize_3d_axes(self):
        """Override: inicializa eixos + reseta handles das elipsoídes."""
        self._ellipsoid_surfaces = []
        super()._initialize_3d_axes()

    # ──────────────────────────────────────────────────────────────────
    # Lógica das elipsoídes
    # ──────────────────────────────────────────────────────────────────

    def _update_ellipsoids(self, frame_number: int):
        """Remove elipsoídes antigas e desenha as do frame atual."""
        # Remove plots anteriores
        for surf in self._ellipsoid_surfaces:
            if surf is not None:
                try:
                    surf.remove()
                except Exception:
                    pass
        self._ellipsoid_surfaces = []

        if self._ellipsoids_per_frame is None or not self._show_ellipsoids:
            return

        # Usa o frame exato se disponível e tiver dados úteis;
        # caso contrário, busca o vizinho mais próximo com pelo menos 1 elipsoíde válida
        raw = self._ellipsoids_per_frame[frame_number] if frame_number < len(self._ellipsoids_per_frame) else None
        has_useful_data = (
            raw is not None
            and isinstance(raw, list)
            and any(e is not None for e in raw)
        )

        if not has_useful_data:
            # Procura o frame computado mais próximo com dado real (até ±100 frames)
            best_idx = None
            for delta in range(1, 101):
                for candidate in (frame_number - delta, frame_number + delta):
                    if 0 <= candidate < len(self._ellipsoids_per_frame):
                        candidate_data = self._ellipsoids_per_frame[candidate]
                        if (
                            candidate_data is not None
                            and isinstance(candidate_data, list)
                            and any(e is not None for e in candidate_data)
                        ):
                            best_idx = candidate
                if best_idx is not None:
                    break
            if best_idx is None:
                return
            frame_data = self._ellipsoids_per_frame[best_idx]
        else:
            frame_data = raw

        n_valid = 0
        n_filtered = 0
        n_error = 0

        for pt_idx, ell in enumerate(frame_data):
            if ell is None:
                self._ellipsoid_surfaces.append(None)
                continue

            # Tenta usar a posição real do ponto no frame atual como centro.
            if hasattr(self, "_skeleton_3d_frame_marker_xyz") and self._skeleton_3d_frame_marker_xyz is not None:
                center = self._skeleton_3d_frame_marker_xyz[frame_number, pt_idx, :]
                # Se o ponto atual é NaN, não desenha a elipsoide
                if np.any(np.isnan(center)):
                    self._ellipsoid_surfaces.append(None)
                    continue
            else:
                center = ell["center"]

            radii = ell["radii"]
            evecs = ell["evecs"]

            # Não plota se NaN ou elipsoíde degenerada
            if np.any(np.isnan(radii)) or np.any(radii < 1e-6):
                n_filtered += 1
                self._ellipsoid_surfaces.append(None)
                continue

            try:
                x, y, z = _make_ellipsoid_surface(center, radii, evecs, n=12)
                surf = self._3d_axes.plot_surface(
                    x, y, z,
                    rstride=1, cstride=1,
                    color=self._ellipsoid_color,
                    alpha=self._ellipsoid_alpha,
                    linewidth=0,
                    antialiased=False,
                    shade=True,
                )
                self._ellipsoid_surfaces.append(surf)
                n_valid += 1
            except Exception as e:
                logger.warning(f"Erro ao plotar elipsoíde: {e}")
                n_error += 1
                self._ellipsoid_surfaces.append(None)

        # Log de diagnóstico apenas no primeiro frame para não poluir
        if frame_number == 0:
            sample = frame_data[0] if frame_data and frame_data[0] is not None else None
            logger.info(
                f"[Elipsoídes frame {frame_number}] válidas={n_valid} filtradas={n_filtered} erros={n_error} "
                f"total={len(frame_data)}"
            )
            if sample is not None:
                logger.info(
                    f"  Exemplo ponto[0]: center={sample['center']} "
                    f"radii={sample['radii']} "
                    f"axes_xlim={self._3d_axes.get_xlim()}"
                )

        self._figure_widget.figure.canvas.draw_idle()
