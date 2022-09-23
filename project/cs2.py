from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtOpenGL import QGLWidget
import sys
from vtk import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk


class vtkMW(QMainWindow):
    """docstring for Mainwindow"""

    def __init__(self, parent=None):
        super(vtkMW, self).__init__(parent)
        self.basic()
        vll = self.kuangti()
        self.setCentralWidget(vll)

    # 窗口基础属性
    def basic(self):
        # 设置标题，大小，图标
        self.setWindowTitle("GT1")
        self.resize(1100, 650)
        self.setWindowIcon(QIcon("renlian.png"))

    def kuangti(self):
        frame = QFrame()
        vl = QVBoxLayout()
        vtkWidget = QVTKRenderWindowInteractor()
        vl.addWidget(vtkWidget)
        # vl.setContentsMargins(0,0,0,0)
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.01, 0.2, 0.01)
        vtkWidget.GetRenderWindow().AddRenderer(ren)
        self.iren = vtkWidget.GetRenderWindow().GetInteractor()
        self.Creatobj(ren)
        self.iren.Initialize()
        frame.setLayout(vl)
        return frame

    def Creatobj(self, ren):
        # Create source
        filename = "BFM2017.obj"
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        ren.AddActor(actor)
        ren.ResetCamera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = vtkMW()
    win.show()
    # win.iren.Initialize()
    sys.exit(app.exec_())
