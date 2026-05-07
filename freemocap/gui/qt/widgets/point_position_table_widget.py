import logging
import numpy as np
from typing import List, Optional
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView, QCheckBox

logger = logging.getLogger(__name__)

class PointPositionTableWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        # Add filtering toggle
        self.centers_checkbox = QCheckBox("Show Centers Only")
        self.centers_checkbox.toggled.connect(self._handle_centers_checkbox_toggled)
        self._layout.addWidget(self.centers_checkbox)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Point Name", "X (m)", "Y (m)", "Z (m)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        
        # Styles for background
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #eeeeee;
                color: #1d1d1f;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #f6e0f5;
                color: #333333;
            }
        """)
        
        self._layout.addWidget(self.table)
        
        self.data: Optional[np.ndarray] = None
        self.point_names: Optional[List[str]] = None
        self.current_frame_index = 0
        
    def set_data(self, data: np.ndarray, point_names: List[str]):
        """
        Set the 3D data and point names.
        data shape: (num_frames, num_points, 3)
        """
        self.data = data
        self.point_names = point_names
        logger.debug(f"PointPositionTableWidget received data with shape {data.shape} and {len(point_names)} point names")
        self.update_table(self.current_frame_index)
        
    def _handle_centers_checkbox_toggled(self):
        self.update_table(self.current_frame_index)

    @Slot(int)
    def update_table(self, frame_index: int):
        """
        Update the table with data from the specified frame index.
        """
        self.current_frame_index = frame_index
        
        if self.data is None or self.point_names is None:
            return
            
        if frame_index < 0 or frame_index >= self.data.shape[0]:
            # logger.warning(f"Frame index {frame_index} out of bounds for data with {self.data.shape[0]} frames")
            return
            
        frame_data = self.data[frame_index]
        
        # Filter logic
        show_centers_only = self.centers_checkbox.isChecked()
        
        filtered_indices = []
        if show_centers_only:
            for i, name in enumerate(self.point_names):
                if 'center' in name.lower():
                    filtered_indices.append(i)
        else:
            filtered_indices = list(range(min(len(self.point_names), frame_data.shape[0])))
            
        self.table.setRowCount(len(filtered_indices))
        
        for row_idx, data_idx in enumerate(filtered_indices):
            name = self.point_names[data_idx]
            self.table.setItem(row_idx, 0, QTableWidgetItem(name))
            
            # Extract X, Y, Z and convert to meters (from mm)
            x, y, z = frame_data[data_idx] / 1000.0
            
            # Create items with formatted strings
            self.table.setItem(row_idx, 1, QTableWidgetItem(f"{x: .4f}"))
            self.table.setItem(row_idx, 2, QTableWidgetItem(f"{y: .4f}"))
            self.table.setItem(row_idx, 3, QTableWidgetItem(f"{z: .4f}"))
            
            # Align text to center for coordinates
            for col in range(1, 4):
                self.table.item(row_idx, col).setTextAlignment(Qt.AlignCenter)

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    widget = PointPositionTableWidget()
    
    # Dummy data
    dummy_data = np.random.rand(100, 5, 3)
    dummy_names = ["tag0_center", "tag0_corner0", "tag1_center", "tag1_corner0", "other_point"]
    widget.set_data(dummy_data, dummy_names)
    
    widget.show()
    sys.exit(app.exec())
