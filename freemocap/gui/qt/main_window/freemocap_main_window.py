import logging
import multiprocessing
import shutil
import threading
import numpy as np
from pathlib import Path
from typing import Union, List, Callable

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QFileDialog,
    QWidget,
    QHBoxLayout,
    QSlider,
)
from PySide6.QtCore import Qt, Slot, QTimer, QThread, Signal
from skelly_viewer import SkellyViewer
from skelly_viewer.utilities.mediapipe_skeleton_builder import build_skeleton, mediapipe_indices, mediapipe_connections
from freemocap.gui.qt.widgets.skeleton_view_with_ellipsoids import SkeletonViewWithEllipsoids
from freemocap.core_processes.capture_volume_calibration.uncertainty_ellipsoid import compute_uncertainty_for_all_frames
from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration.get_anipose_calibration_object import load_anipose_calibration_toml_from_path
from skellycam import (
    SkellyCamParameterTreeWidget,
    SkellyCamWidget,
)
from tqdm import tqdm

from freemocap.data_layer.generate_jupyter_notebook.generate_jupyter_notebook import (
    generate_jupyter_notebook,
)
from freemocap.data_layer.recording_models.post_processing_parameter_models import (
    ProcessingParameterModel,
)
from freemocap.data_layer.recording_models.recording_info_model import (
    RecordingInfoModel,
)
from freemocap.gui.qt.actions_and_menus.actions import Actions
from freemocap.gui.qt.actions_and_menus.menu_bar import MenuBar
from freemocap.gui.qt.style_sheet.css_file_watcher import CSSFileWatcher
from freemocap.gui.qt.style_sheet.scss_file_watcher import SCSSFileWatcher
from freemocap.gui.qt.style_sheet.set_css_style_sheet import apply_css_style_sheet
from freemocap.gui.qt.utilities.copy_timestamps_folder import copy_directory_if_contains_timestamps
from freemocap.gui.qt.utilities.get_qt_app import get_qt_app
from freemocap.gui.qt.utilities.save_and_load_gui_state import (
    GuiState,
    load_gui_state,
    save_gui_state,
)
from freemocap.gui.qt.utilities.update_most_recent_recording_toml import (
    update_most_recent_recording_toml,
)
from freemocap.gui.qt.widgets.active_recording_widget import ActiveRecordingInfoWidget
from freemocap.gui.qt.widgets.camera_controller_group_box import CameraControllerGroupBox
from freemocap.gui.qt.widgets.central_tab_widget import CentralTabWidget
from freemocap.gui.qt.widgets.control_panel.control_panel_dock_widget import (
    ControlPanelWidget,
)
from freemocap.gui.qt.widgets.control_panel.export_data_control_panel import VisualizationControlPanel
from freemocap.gui.qt.widgets.control_panel.process_mocap_data_panel.process_motion_capture_data_panel import (
    ProcessMotionCaptureDataPanel,
)
from freemocap.gui.qt.widgets.directory_view_widget import DirectoryViewWidget
from freemocap.gui.qt.widgets.home_widget import (
    HomeWidget,
)
from freemocap.gui.qt.widgets.point_position_table_widget import PointPositionTableWidget
from freemocap.gui.qt.widgets.import_videos_wizard import ImportVideosWizard
from freemocap.gui.qt.widgets.log_view_widget import LogViewWidget
from freemocap.gui.qt.widgets.opencv_conflict_dialog import OpencvConflictDialog
from freemocap.gui.qt.widgets.release_notes_dialogs.tabbed_release_notes_dialog import TabbedReleaseNotesDialog
from freemocap.gui.qt.widgets.set_data_folder_dialog import SetDataFolderDialog
from freemocap.gui.qt.widgets.welcome_screen_dialog import WelcomeScreenDialog
from freemocap.gui.qt.workers.download_sample_data_thread_worker import DownloadDataThreadWorker
from freemocap.gui.qt.workers.export_to_blender_thread_worker import ExportToBlenderThreadWorker
# reboot GUI method based on this - https://stackoverflow.com/a/56563926/14662833
from freemocap.system.open_file import open_file
from freemocap.system.paths_and_filenames.file_and_folder_names import (
    PATH_TO_FREEMOCAP_LOGO_SVG,
)
from freemocap.system.paths_and_filenames.path_getters import (
    get_recording_session_folder_path,
    get_css_stylesheet_path,
    get_scss_stylesheet_path,
    get_blender_file_path,
    get_most_recent_recording_path,
    get_gui_state_json_path,
)
from freemocap.system.user_data.pipedream_pings import PipedreamPings
from freemocap.utilities.remove_empty_directories import remove_empty_directories

EXIT_CODE_REBOOT = -123456789

logger = logging.getLogger(__name__)


class UncertaintyWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, cgroup, image_2d_data, sigma_pixels, subsample_frames):
        super().__init__()
        self.cgroup = cgroup
        self.image_2d_data = image_2d_data
        self.sigma_pixels = sigma_pixels
        self.subsample_frames = subsample_frames

    def run(self):
        try:
            from freemocap.core_processes.capture_volume_calibration.uncertainty_ellipsoid import compute_uncertainty_for_all_frames
            ellipsoids = compute_uncertainty_for_all_frames(
                cgroup=self.cgroup,
                image_2d_data=self.image_2d_data,
                sigma_pixels=self.sigma_pixels,
                subsample_frames=self.subsample_frames,
            )
            self.finished.emit(ellipsoids)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(
            self,
            freemocap_data_folder_path: Union[str, Path],
            pipedream_pings: PipedreamPings,
            parent=None,
    ):
        super().__init__(parent=parent)
        self._log_view_widget = LogViewWidget(parent=self)  # start this first so it will grab the setup logs
        logger.info("Initializing FreeMoCap MainWindow")

        self.setMinimumSize(1280, 720)
        self.setWindowIcon(QIcon(PATH_TO_FREEMOCAP_LOGO_SVG))
        self.setWindowTitle("freemocap \U0001f480 \U00002728")

        dummy_widget = QWidget()
        self._layout = QHBoxLayout()
        dummy_widget.setLayout(self._layout)
        self.setCentralWidget(dummy_widget)

        self._css_file_watcher = self._set_up_stylesheet()

        self._freemocap_data_folder_path = freemocap_data_folder_path
        self._pipedream_pings = pipedream_pings

        self._gui_state: GuiState = load_gui_state(get_gui_state_json_path())

        self._kill_thread_event = multiprocessing.Event()

        self._active_recording_info_widget = ActiveRecordingInfoWidget(parent=self)
        self._active_recording_info_widget.new_active_recording_selected_signal.connect(
            self._handle_new_active_recording_selected
        )
        self._directory_view_widget = self._create_directory_view_widget()
        self._directory_view_widget.expand_directory_to_path(get_recording_session_folder_path())
        self._directory_view_widget.new_active_recording_selected_signal.connect(
            self._active_recording_info_widget.set_active_recording
        )

        self._actions = Actions(freemocap_main_window=self)

        self._menu_bar = MenuBar(actions=self._actions, parent=self)
        self.setMenuBar(self._menu_bar)

        self.statusBar().showMessage(
            "Watch the terminal output for status updates, we're working on integrating better status updates into the GUI"
        )

        self._central_tab_widget = self._create_central_tab_widget()
        self.setCentralWidget(self._central_tab_widget)

        self._tools_dock_widget = self._create_tools_dock_widget()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._tools_dock_widget)

        self._control_panel_widget = self._create_control_panel_widget(log_update=self._log_view_widget.add_log)
        self._tools_dock_widget.setWidget(self._control_panel_widget)

        self._connect_calibration_updates()

        log_view_dock_widget = QDockWidget("Log View", self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_view_dock_widget)
        log_view_dock_widget.setWidget(self._log_view_widget)
        log_view_dock_widget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        logger.debug("Finished initializing FreeMoCap MainWindow")

    def _create_tools_dock_widget(self):
        tools_dock_widget = QDockWidget("Control Panel", self)
        tools_dock_widget.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        tools_dock_widget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        return tools_dock_widget

    def _handle_videos_saved_to_this_folder_signal(self, folder_path: str):
        logger.debug(f"Videos saved to this folder signal received: {folder_path}")

        self._active_recording_info_widget.set_active_recording(recording_folder_path=folder_path)

        if (
                self._controller_group_box.auto_process_videos_checked
                and self._controller_group_box.mocap_videos_radio_button_checked
        ):
            logger.info("'Auto process videos' checkbox is checked - triggering 'Process Motion Capture Data' button")
            self._process_motion_capture_data_panel.process_motion_capture_data_button.click()
        elif self._controller_group_box.calibration_videos_radio_button_checked:
            logger.info("Processing calibration videos")
            self._process_motion_capture_data_panel.calibrate_from_active_recording(
                charuco_square_size_mm=float(self._controller_group_box.charuco_square_size),
                use_charuco_as_groundplane=self._controller_group_box.use_charuco_as_groundplane,
                charuco_board_name=self._controller_group_box.charuco_board_name,
            )

    def _handle_processing_finished_signal(self):
        # Update the active tracker in the recording info model from the panel's selection
        session_params = self._process_motion_capture_data_panel._create_session_parameter_model()
        self._active_recording_info_widget.active_recording_info.active_tracker = session_params.tracking_model_info.name
        
        logger.info("Processamento de dados concluído. Agendando atualização da interface e elipsoídes...")
        
        # O atraso ajuda a garantir que a thread de processamento anterior 
        # tenha tempo de fechar completamente antes de começarmos cálculos pesados
        QTimer.singleShot(500, self._update_skelly_viewer_widget)
        
        if self._controller_group_box.auto_open_in_blender_checked and not self._kill_thread_event.is_set():
            logger.info("'Auto Open in Blender' checkbox is checked - triggering 'Create Blender Scene'")
            self._export_active_recording_to_blender()
        if self._controller_group_box.generate_jupyter_notebook_checked and not self._kill_thread_event.is_set():
            self._generate_jupyter_notebook()
        logger.info("Processing finished")

    def handle_start_new_session_action(self):
        # self._central_tab_widget.set_welcome_tab_enabled(True)
        self._central_tab_widget.set_camera_view_tab_enabled(True)
        self._central_tab_widget.setCurrentIndex(1)
        self._controller_group_box.show()
        self._skellycam_widget.detect_available_cameras()

    def update(self):
        super().update()

        try:
            if not self._skellycam_widget.is_recording:
                self._controller_group_box.update_recording_name_string()
        except Exception as e:
            logger.exception(e)

    def _set_up_stylesheet(self):
        apply_css_style_sheet(self, get_css_stylesheet_path())
        SCSSFileWatcher(
            path_to_scss_file=get_scss_stylesheet_path(), path_to_css_file=get_css_stylesheet_path(), parent=self
        )
        css_file_watcher = CSSFileWatcher(path_to_css_file=get_css_stylesheet_path(), parent=self)
        return css_file_watcher

    def _create_new_synchronized_videos_folder(self) -> str:
        new_recording_folder_path = self._controller_group_box.get_new_recording_path()
        logger.info(f"Creating new recording folder at: {new_recording_folder_path}")
        self._active_recording_info_widget.set_active_recording(recording_folder_path=new_recording_folder_path)
        return self._active_recording_info_widget.active_recording_info.synchronized_videos_folder_path

    def _create_central_tab_widget(self):
        self._home_widget = HomeWidget(actions=self._actions, gui_state=self._gui_state, parent=self)

        self._skellycam_widget = SkellyCamWidget(
            self._create_new_synchronized_videos_folder,
            parent=self,
        )
        self._skellycam_widget.videos_saved_to_this_folder_signal.connect(
            self._handle_videos_saved_to_this_folder_signal
        )

        self._controller_group_box = CameraControllerGroupBox(
            skellycam_widget=self._skellycam_widget, gui_state=self._gui_state, parent=self
        )

        self._skelly_viewer_widget = SkellyViewer()

        # Swap SkeletonViewWidget with SkeletonViewWithEllipsoids inside the Qt layout.
        # Just reassigning _skeleton_view_widget is NOT enough — the old widget stays
        # in the layout and the new one is never displayed. We must replace it in-place.
        old_sv = self._skelly_viewer_widget._skeleton_view_widget
        new_sv = SkeletonViewWithEllipsoids()
        new_sv.setFixedSize(old_sv.size())

        def _replace_widget_in_layout(layout, old_w, new_w) -> bool:
            """Recursively search nested layouts and replace old_w with new_w."""
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget() is old_w:
                    layout.replaceWidget(old_w, new_w)
                    old_w.setParent(None)
                    return True
                if item.layout():
                    if _replace_widget_in_layout(item.layout(), old_w, new_w):
                        return True
            return False

        replaced = _replace_widget_in_layout(self._skelly_viewer_widget.layout(), old_sv, new_sv)
        if not replaced:
            logger.warning("Could not replace SkeletonViewWidget in layout — ellipsoids may not be visible")

        self._skelly_viewer_widget._skeleton_view_widget = new_sv

        # Reconnect the signal that SkellyViewer uses to react to new data
        new_sv.skeleton_data_loaded_signal.connect(
            self._skelly_viewer_widget._handle_data_loaded_signal
        )

        self._point_position_table_widget = PointPositionTableWidget(parent=self)

        # Connect slider signal
        slider = self._skelly_viewer_widget.findChild(QSlider)
        if slider:
            slider.valueChanged.connect(self._point_position_table_widget.update_table)
        else:
            logger.warning("Could not find frame slider in SkellyViewer")

        center_tab_widget = CentralTabWidget(
            parent=self,
            skelly_cam_widget=self._skellycam_widget,
            camera_controller_widget=self._controller_group_box,
            welcome_to_freemocap_widget=self._home_widget,
            skelly_viewer_widget=self._skelly_viewer_widget,
            point_position_table_widget=self._point_position_table_widget,
            directory_view_widget=self._directory_view_widget,
            active_recording_info_widget=self._active_recording_info_widget,
        )

        center_tab_widget.set_welcome_tab_enabled(True)
        center_tab_widget.set_camera_view_tab_enabled(True)
        center_tab_widget.set_visualize_data_tab_enabled(True)

        return center_tab_widget

    def _create_directory_view_widget(self):
        return DirectoryViewWidget(
            gui_state=self._gui_state,
            get_active_recording_info_callable=self._active_recording_info_widget.get_active_recording_info,
        )

    def _create_control_panel_widget(self, log_update: Callable):
        self._camera_configuration_parameter_tree_widget = SkellyCamParameterTreeWidget(self._skellycam_widget)

        self._process_motion_capture_data_panel = ProcessMotionCaptureDataPanel(
            recording_processing_parameters=ProcessingParameterModel(),
            get_active_recording_info=self._active_recording_info_widget.get_active_recording_info,
            gui_state=self._gui_state,
            kill_thread_event=self._kill_thread_event,
            log_update=log_update,
        )
        self._process_motion_capture_data_panel.processing_finished_signal.connect(
            self._handle_processing_finished_signal
        )

        self._visualization_control_panel = VisualizationControlPanel(parent=self, gui_state=self._gui_state)
        self._visualization_control_panel.export_to_blender_button.clicked.connect(
            self._export_active_recording_to_blender
        )

        self._visualization_control_panel.generate_jupyter_notebook_button.clicked.connect(
            self._generate_jupyter_notebook
        )

        return ControlPanelWidget(
            camera_configuration_parameter_tree_widget=self._camera_configuration_parameter_tree_widget,
            process_motion_capture_data_panel=self._process_motion_capture_data_panel,
            visualize_data_widget=self._visualization_control_panel,
            parent=self,
        )

    def _export_active_recording_to_blender(self):
        logger.debug("Exporting active recording to Blender...")
        recording_path = self._active_recording_info_widget.get_active_recording_path()

        if self._visualization_control_panel.blender_executable_path is None:
            logger.error("Blender executable path is None!")
            return

        if not recording_path:
            logger.error("Recording path is None!")
            return

        self._export_to_blender_thread_worker = ExportToBlenderThreadWorker(
            recording_path=recording_path,
            blender_file_path=Path(get_blender_file_path(recording_path)),
            blender_executable_path=Path(self._visualization_control_panel.blender_executable_path),
            kill_thread_event=self._kill_thread_event,
        )
        self._export_to_blender_thread_worker.start()
        self._export_to_blender_thread_worker.success.connect(self._handle_export_to_blender_finished)

    @Slot()
    def _handle_export_to_blender_finished(self, success_value: bool) -> None:
        if success_value is False:
            logger.error("Blender export failed!")
        elif self._controller_group_box.auto_open_in_blender_checked:
            if Path(self._active_recording_info_widget.active_recording_info.blender_file_path).exists():
                open_file(self._active_recording_info_widget.active_recording_info.blender_file_path)
            else:
                logger.error(
                    "Blender file does not exist! Did something go wrong in the `export_to_blender` call above?"
                )

    def _generate_jupyter_notebook(self):
        logger.info("Exporting active recording to a Jupyter notebook...")
        recording_path = self._active_recording_info_widget.get_active_recording_path()
        # TODO: Need to include jupyter notebook in recording files that we keep track of (2023-05-15)
        if recording_path:
            generate_jupyter_notebook(path_to_recording=recording_path)

    def _handle_new_active_recording_selected(self, recording_info_model: RecordingInfoModel):
        logger.info(f"New active recording selected: {recording_info_model.path}")

        self._pipedream_pings.update_pings_dict(
            key=recording_info_model.name, value=self._active_recording_info_widget.active_recording_info.status_check
        )

        path = Path(recording_info_model.path)
        path.mkdir(parents=True, exist_ok=True)

        if str(path.parent) != get_recording_session_folder_path():
            self._directory_view_widget.set_folder_as_root(path.parent)
        else:
            self._directory_view_widget.set_folder_as_root(path)

        if Path(recording_info_model.synchronized_videos_folder_path).exists():
            self._directory_view_widget.expand_directory_to_path(recording_info_model.synchronized_videos_folder_path)
        else:
            self._directory_view_widget.expand_directory_to_path(recording_info_model.path)

        self._active_recording_info_widget.update_parameter_tree()
        # self._recording_name_label.setText(f"Recording Name: {recording_info_model.name}")
        self._update_skelly_viewer_widget()
        self._directory_view_widget.handle_new_active_recording_selected()

        try:
            self._process_motion_capture_data_panel.update_calibration_path()
        except (
                AttributeError
        ):  # Active Recording and Data Panel widgets rely on each other, so we're guaranteed to hit this every time the app opens
            logger.debug("Process motion capture data panel not created yet, skipping claibraiton setting")
        except Exception as e:
            logger.error(e)

        update_most_recent_recording_toml(recording_info_model=recording_info_model)

    def _update_skelly_viewer_widget(self):
        active_recording_info = self._active_recording_info_widget.active_recording_info

        if active_recording_info.data3d_status_check:
            self._load_generic_skeleton_data(
                skelly_viewer=self._skelly_viewer_widget,
                npy_path=active_recording_info.data_3d_npy_file_path,
                recording_info=active_recording_info
            )

        if active_recording_info.data2d_status_check:
            self._skelly_viewer_widget.generate_video_display(
                video_folder_path=active_recording_info.annotated_videos_folder_path
            )

        elif active_recording_info.synchronized_videos_status_check:
            self._skelly_viewer_widget.generate_video_display(
                video_folder_path=active_recording_info.synchronized_videos_folder_path
            )

    def _load_generic_skeleton_data(self, skelly_viewer: SkellyViewer, npy_path: Union[str, Path], recording_info: RecordingInfoModel):
        try:
            data = np.load(str(npy_path))
        except Exception as e:
            logger.error(f"Could not load skeleton data from {npy_path}: {e}")
            return

        # Default to Mediapipe settings
        indices = mediapipe_indices
        connections = mediapipe_connections
        
        # Try to detect if it's an AprilTag model based on shape or name
        num_points = data.shape[1]
        is_apriltag = (recording_info.active_tracker == 'apriltag') or (num_points < 33)

        if is_apriltag:
            try:
                # Reconstruct model info based on data shape
                if num_points % 5 == 0:
                    # Likely point_mode='both' (5 points per tag)
                    num_tags = num_points // 5
                    tag_ids = tuple(range(num_tags))
                    from skellytracker.trackers.apriltag_tracker.apriltag_model_info import (
                        april_tag_landmark_names, 
                        april_tag_segment_connections
                    )
                    indices = april_tag_landmark_names(tag_ids, point_mode='both')
                    raw_connections = april_tag_segment_connections(tag_ids, point_mode='both')
                    connections = {name: [val['proximal'], val['distal']] for name, val in raw_connections.items()}
                elif num_points % 4 == 0:
                    # Likely corners
                    num_tags = num_points // 4
                    tag_ids = tuple(range(num_tags))
                    from skellytracker.trackers.apriltag_tracker.apriltag_model_info import (
                        april_tag_landmark_names, 
                        april_tag_segment_connections
                    )
                    indices = april_tag_landmark_names(tag_ids, point_mode='corners')
                    raw_connections = april_tag_segment_connections(tag_ids, point_mode='corners')
                    connections = {name: [val['proximal'], val['distal']] for name, val in raw_connections.items()}
                elif num_points == 1:
                    # Single center
                    tag_ids = (0,)
                    from skellytracker.trackers.apriltag_tracker.apriltag_model_info import april_tag_landmark_names
                    indices = april_tag_landmark_names(tag_ids, point_mode='center')
                    connections = {}
            except Exception as e:
                logger.warning(f"Could not resolve dynamic AprilTag model info, falling back to defaults: {e}")

        # Manually perform the work of load_skeleton_data but with our dynamic indices/connections
        skeleton_view_widget = skelly_viewer._skeleton_view_widget
        skeleton_view_widget._skeleton_3d_frame_marker_xyz = data
        
        try:
            skeleton_view_widget._mediapipe_skeleton = build_skeleton(
                skeleton_3d_frame_marker_xyz=data,
                pose_estimation_markers_list=indices,
                pose_estimation_connections_dict=connections
            )
            skeleton_view_widget._number_of_frames = data.shape[0]
            skeleton_view_widget._initialize_3d_axes()
            skeleton_view_widget.skeleton_data_loaded_signal.emit()

            self._point_position_table_widget.set_data(data, indices)
        except Exception as e:
            logger.error(f"Failed to build skeleton for viewer: {e}")

        # Calcula e exibe as elipsoídes de incerteza (em thread separada para não travar a GUI)
        try:
            self._compute_and_load_ellipsoids(recording_info)
        except Exception as e:
            logger.warning(f"Não foi possível calcular elipsoídes de incerteza: {e}")

    def _compute_and_load_ellipsoids(self, recording_info):
        """
        Carrega a calibração e os dados 2D, e inicia o cálculo das elipsoídes
        em uma thread separada para não travar a GUI.
        """
        calibration_path = recording_info.calibration_toml_path
        if not calibration_path or not Path(calibration_path).exists():
            logger.warning("Arquivo de calibração não encontrado — elipsoídes não serão calculadas.")
            return

        # Encontra o arquivo de dados 2D usando o padrão de nomenclatura do freemocap
        from freemocap.system.paths_and_filenames.file_and_folder_names import DATA_2D_NPY_FILE_NAME, OLD_DATA_2D_NPY_FILE_NAME
        raw_data_path = Path(recording_info.raw_data_folder_path)
        tracker_prefix = getattr(recording_info, "active_tracker", "") or ""

        # Tenta o arquivo com prefixo do tracker, depois sem prefixo, depois o formato antigo
        candidates = [
            raw_data_path / f"{tracker_prefix}_{DATA_2D_NPY_FILE_NAME}",
            raw_data_path / DATA_2D_NPY_FILE_NAME,
            raw_data_path / OLD_DATA_2D_NPY_FILE_NAME,
        ]
        npy_2d_path = next((p for p in candidates if p.exists()), None)
        if npy_2d_path is None:
            npy_2d_files = list(raw_data_path.glob("*2dData*.npy"))
            npy_2d_path = npy_2d_files[0] if npy_2d_files else None
        
        if npy_2d_path is None:
            logger.warning("Arquivo de dados 2D não encontrado.")
            return
        logger.info(f"Calculando elipsoídes de incerteza usando: {npy_2d_path.name}")

        image_2d_data = np.load(str(npy_2d_path))
        # Garante shape [n_cams, n_frames, n_points, 2]
        if image_2d_data.ndim == 4 and image_2d_data.shape[-1] >= 2:
            image_2d_data = image_2d_data[:, :, :, :2]
        else:
            return

        cgroup = load_anipose_calibration_toml_from_path(calibration_path)

        # Se já houver um cálculo rodando, para ele antes de começar o novo
        if hasattr(self, "_uncertainty_thread") and self._uncertainty_thread.isRunning():
            logger.info("Cancelando cálculo de incerteza anterior...")
            self._uncertainty_thread.finished.disconnect()
            self._uncertainty_thread.terminate()
            self._uncertainty_thread.wait()

        # Inicia o cálculo em thread separada
        logger.info(f"Iniciando cálculo de incerteza (todos os frames) em background...")
        self._uncertainty_thread = UncertaintyWorker(
            cgroup=cgroup,
            image_2d_data=image_2d_data,
            sigma_pixels=3.0,
            subsample_frames=1,
        )
        self._uncertainty_thread.finished.connect(self._handle_uncertainty_finished)
        self._uncertainty_thread.error.connect(lambda e: logger.warning(f"Erro no cálculo de incerteza: {e}"))
        self._uncertainty_thread.start()

    @Slot(list)
    def _handle_uncertainty_finished(self, ellipsoids):
        skeleton_view = self._skelly_viewer_widget._skeleton_view_widget
        if hasattr(skeleton_view, "set_uncertainty_ellipsoids"):
            skeleton_view.set_uncertainty_ellipsoids(ellipsoids)
            logger.info("Elipsoídes de incerteza calculadas e carregadas no viewer.")
            
            # Redesenha o frame atual
            try:
                slider = self._skelly_viewer_widget.findChild(QSlider)
                current_frame = slider.value() if slider is not None else 0
                skeleton_view.update_skeleton_plot(current_frame)
            except Exception:
                pass
        else:
            logger.warning("O viewer 3D não suporta elipsoídes.")


    def kill_running_threads_and_processes(self):
        logger.info("Killing running threads and processes... ")
        try:
            self._skellycam_widget.close()
        except Exception as e:
            logger.error(f"Error killing running threads and processes: {e}")

        # Finaliza a thread de incerteza se ela existir
        if hasattr(self, "_uncertainty_thread") and self._uncertainty_thread.isRunning():
            logger.info("Finalizando thread de incerteza...")
            self._uncertainty_thread.terminate()
            self._uncertainty_thread.wait()

        self._kill_thread_event.set()

    def handle_load_most_recent_recording(self):
        logger.info("`Load Most Recent Recording` QAction triggered")
        most_recent_recording_path = get_most_recent_recording_path()

        if most_recent_recording_path is None:
            logger.error("`get_most_recent_recording_path()` return `None`!")
            return

        self._active_recording_info_widget.set_active_recording(recording_folder_path=get_most_recent_recording_path())
        self._central_tab_widget.setCurrentIndex(2)
        self._control_panel_widget.tab_widget.setCurrentWidget(self._process_motion_capture_data_panel)

    def open_load_existing_recording_dialog(self):
        # from this tutorial - https://www.youtube.com/watch?v=gg5TepTc2Jg&t=649s
        logger.info("Opening `Load Existing Recording` dialog... ")
        user_selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Select a recording folder",
            str(get_recording_session_folder_path()),
        )
        if len(user_selected_directory) == 0:
            logger.info("User cancelled `Load Existing Recording` dialog")
            return

        logger.info(f"User selected recording path:{user_selected_directory}")

        self._active_recording_info_widget.set_active_recording(recording_folder_path=user_selected_directory)
        self._central_tab_widget.setCurrentIndex(2)

    def reset_to_default_gui_settings(self):
        self._gui_state = GuiState()

        self._home_widget._send_pings_checkbox.setChecked(self._gui_state.send_user_pings)
        self._controller_group_box._auto_process_videos_checkbox.setChecked(self._gui_state.auto_process_videos_on_save)
        self._controller_group_box._generate_jupyter_notebook_checkbox.setChecked(
            self._gui_state.generate_jupyter_notebook
        )
        self._controller_group_box._auto_open_in_blender_checkbox.setChecked(self._gui_state.auto_open_in_blender)
        self._controller_group_box._charuco_square_size_line_edit.setText(str(self._gui_state.charuco_square_size))
        self._process_motion_capture_data_panel._calibration_control_panel._charuco_square_size_line_edit.setText(
            str(self._gui_state.charuco_square_size)
        )
        self._visualization_control_panel._blender_executable_label.setText(str(self._gui_state.blender_path))
        self._visualization_control_panel._blender_executable_path = str(self._gui_state.blender_path)

        save_gui_state(self._gui_state)

        self._active_recording_info_widget.set_active_recording(recording_folder_path=get_most_recent_recording_path())

    def open_import_videos_dialog(self):
        # from this tutorial - https://www.youtube.com/watch?v=gg5TepTc2Jg&t=649s
        logger.info("Opening `Import Videos` dialog... ")

        import_videos_path = QFileDialog.getExistingDirectory(
            self,
            "Select a folder containing synchronized videos (each video must have *exactly* the same number of frames)",
            str(Path.home()),
        )

        if len(import_videos_path) == 0:
            logger.info("User cancelled `Import Videos` dialog")
            return

        self._import_videos_window = ImportVideosWizard(
            parent=self,
            import_videos_path=import_videos_path,
            kill_thread_event=self._kill_thread_event,
        )
        self._import_videos_window.folder_to_save_videos_to_selected.connect(self._handle_import_videos)
        self._import_videos_window.exec()

    def open_welcome_screen_dialog(self):
        logger.info("Opening `Welcome to Freemocap` dialog... ")

        self._welcome_screen_dialog = WelcomeScreenDialog(
            gui_state=self._gui_state, kill_thread_event=self._kill_thread_event, parent=self
        )

        self._welcome_screen_dialog.exec()

    def open_release_notes_popup(self):
        logger.info("Opening `Release Notes` dialog... ")

        self._gui_state.shown_latest_release_notes = True

        save_gui_state(gui_state=self._gui_state, file_pathstring=get_gui_state_json_path())

        dialog = TabbedReleaseNotesDialog(
            kill_thread_event=threading.Event(),
            gui_state=self._gui_state,
            dark_mode=True,  # Set to True for dark mode, False for light mode
        )
        dialog.exec()

    def open_opencv_conflict_dialog(self):
        self._opencv_conflict_dialog = OpencvConflictDialog(
            gui_state=self._gui_state, kill_thread_event=self._kill_thread_event, parent=self
        )

        self._opencv_conflict_dialog.exec()

    def open_settings_dialog(self):
        self._settings_dialog = SetDataFolderDialog(
            gui_state=self._gui_state, kill_thread_event=self._kill_thread_event, parent=self
        )

        self._settings_dialog.exec()

        if self._settings_dialog.result():
            self.reboot_gui()

    def download_data(self, dataset_name: str):
        logger.info("Downloading sample data")
        self.download_data_thread_worker = DownloadDataThreadWorker(dataset_name=dataset_name)
        self.download_data_thread_worker.start()
        self.download_data_thread_worker.finished.connect(self._handle_download_data_finished)

    @Slot(str)
    def _handle_download_data_finished(self, downloaded_data_path: str):
        logger.info("Setting downloaded data as active recording... ")
        self._active_recording_info_widget.set_active_recording(recording_folder_path=downloaded_data_path)

    @Slot(list, str, bool)
    def _handle_import_videos(self, video_paths: List[str], folder_to_save_videos: str, synchronization_bool: bool):
        folder_to_save_videos = Path(folder_to_save_videos)
        folder_to_save_videos.mkdir(parents=True, exist_ok=True)

        if len(video_paths) == 0:
            logger.error("No videos to import!")
            return

        if not synchronization_bool:
            for video_path in tqdm(
                    video_paths,
                    desc="Importing videos...",
                    colour=[255, 128, 0],
                    unit="video",
                    unit_scale=True,
                    leave=False,
            ):
                if not Path(video_path).exists():
                    logger.error(f"{video_path} does not exist!")
                    return

                destination_path = folder_to_save_videos / Path(video_path).name
                logger.info(f"Copying video from {video_path} to {destination_path}")

                shutil.copy(video_path, destination_path)

        timestamps_copied = copy_directory_if_contains_timestamps(
            source_dir=Path(video_paths[0]).parent, destination_dir=folder_to_save_videos
        )

        if timestamps_copied:
            logger.info(f"Copied timestamps from {Path(video_paths[0]).parent} to {folder_to_save_videos}")
        else:
            logger.info(f"No timestamps found in {Path(video_paths[0]).parent}")

        self._active_recording_info_widget.set_active_recording(
            recording_folder_path=Path(folder_to_save_videos).parent
        )

    def _connect_calibration_updates(self):
        self._process_motion_capture_data_panel._calibration_control_panel.control_panel_calibration_updated.connect(
            self._controller_group_box.charuco_option_updated
        )
        self._controller_group_box.controller_group_box_calibration_updated.connect(
            self._process_motion_capture_data_panel._calibration_control_panel.charuco_option_updated
        )

    def reboot_gui(self):
        logger.info("Rebooting GUI... ")
        get_qt_app().exit(EXIT_CODE_REBOOT)

    def closeEvent(self, a0) -> None:
        logger.info("Main window `closeEvent` detected")

        if self._home_widget.consent_to_send_usage_information:
            self._pipedream_pings.update_pings_dict(key="gui_closed", value=True)
            if self._active_recording_info_widget.active_recording_info is not None:
                self._pipedream_pings.update_pings_dict(
                    key="active recording status on close",
                    value=self._active_recording_info_widget.active_recording_info.status_check,
                )
            self._pipedream_pings.send_pipedream_ping()

        try:
            remove_empty_directories(get_recording_session_folder_path())
        except Exception as e:
            logger.error(f"Error while removing empty directories: {e}")

        try:
            self._skellycam_widget.close()
        except Exception as e:
            logger.error(f"Error while closing the viewer widget: {e}")
        super().closeEvent(a0)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow(pipedream_pings=PipedreamPings())
    main_window.show()
    app.exec()
    for process in multiprocessing.active_children():
        logger.info(f"Terminating process: {process}")
        process.terminate()
    sys.exit()
