import sys
from PyQt5.QtWidgets import QApplication, QMessageBox
from face_swap_ui_enhanced import EnhancedFaceSwapUI

try:
    app = QApplication(sys.argv)
    print("Step 1: QApplication created")
    
    window = EnhancedFaceSwapUI()
    print("Step 2: Window created")
    
    window.show()
    print("Step 3: Window shown")
    print("Window visible:", window.isVisible())
    
    # Keep the window open
    result = app.exec_()
    print(f"Application exited with code: {result}")
    sys.exit(result)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    QMessageBox.critical(None, "Error", f"Failed to start: {e}")
    sys.exit(1)
