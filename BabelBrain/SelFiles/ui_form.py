# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(1025, 369)
        self.ContinuepushButton = QPushButton(Dialog)
        self.ContinuepushButton.setObjectName(u"ContinuepushButton")
        self.ContinuepushButton.setGeometry(QRect(378, 336, 239, 32))
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(6, 3, 1014, 237))
        self.groupBox.setAutoFillBackground(False)
        self.groupBox.setStyleSheet(u"")
        self.CoregCTlabel = QLabel(self.groupBox)
        self.CoregCTlabel.setObjectName(u"CoregCTlabel")
        self.CoregCTlabel.setEnabled(False)
        self.CoregCTlabel.setGeometry(QRect(150, 134, 58, 16))
        self.SelT1WpushButton = QPushButton(self.groupBox)
        self.SelT1WpushButton.setObjectName(u"SelT1WpushButton")
        self.SelT1WpushButton.setGeometry(QRect(256, 93, 135, 32))
        self.SelT1WpushButton.setStyleSheet(u"")
        self.SelT1WpushButton.setAutoDefault(False)
        self.SelT1WpushButton.setFlat(False)
        self.SelCTpushButton = QPushButton(self.groupBox)
        self.SelCTpushButton.setObjectName(u"SelCTpushButton")
        self.SelCTpushButton.setEnabled(False)
        self.SelCTpushButton.setGeometry(QRect(305, 127, 84, 30))
        self.SelCTpushButton.setStyleSheet(u"")
        self.SelCTpushButton.setAutoDefault(False)
        self.SelCTpushButton.setFlat(False)
        self.SimbNIBSlineEdit = QLineEdit(self.groupBox)
        self.SimbNIBSlineEdit.setObjectName(u"SimbNIBSlineEdit")
        self.SimbNIBSlineEdit.setGeometry(QRect(400, 71, 607, 21))
        self.SimbNIBSlineEdit.setStyleSheet(u"")
        self.SelTProfilepushButton = QPushButton(self.groupBox)
        self.SelTProfilepushButton.setObjectName(u"SelTProfilepushButton")
        self.SelTProfilepushButton.setGeometry(QRect(247, 162, 131, 56))
        font = QFont()
        font.setBold(False)
        self.SelTProfilepushButton.setFont(font)
        self.SelTProfilepushButton.setStyleSheet(u"")
        self.SelTProfilepushButton.setAutoDefault(False)
        self.SelTProfilepushButton.setFlat(False)
        self.ThermalProfilelineEdit = QLineEdit(self.groupBox)
        self.ThermalProfilelineEdit.setObjectName(u"ThermalProfilelineEdit")
        self.ThermalProfilelineEdit.setGeometry(QRect(402, 182, 607, 21))
        self.ThermalProfilelineEdit.setStyleSheet(u"")
        self.CTlineEdit = QLineEdit(self.groupBox)
        self.CTlineEdit.setObjectName(u"CTlineEdit")
        self.CTlineEdit.setEnabled(False)
        self.CTlineEdit.setGeometry(QRect(400, 132, 607, 21))
        self.CTlineEdit.setStyleSheet(u"")
        self.CTlineEdit.setCursorPosition(3)
        self.CTlineEdit.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.T1WlineEdit = QLineEdit(self.groupBox)
        self.T1WlineEdit.setObjectName(u"T1WlineEdit")
        self.T1WlineEdit.setGeometry(QRect(401, 98, 607, 21))
        self.T1WlineEdit.setStyleSheet(u"")
        self.CTTypecomboBox = QComboBox(self.groupBox)
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.setObjectName(u"CTTypecomboBox")
        self.CTTypecomboBox.setGeometry(QRect(70, 130, 82, 30))
        self.CTTypecomboBox.setStyleSheet(u"")
        self.SelSimbNIBSpushButton = QPushButton(self.groupBox)
        self.SelSimbNIBSpushButton.setObjectName(u"SelSimbNIBSpushButton")
        self.SelSimbNIBSpushButton.setGeometry(QRect(255, 66, 135, 32))
        self.SelSimbNIBSpushButton.setStyleSheet(u"")
        self.SelSimbNIBSpushButton.setAutoDefault(False)
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 135, 58, 16))
        self.SelTrajectorypushButton = QPushButton(self.groupBox)
        self.SelTrajectorypushButton.setObjectName(u"SelTrajectorypushButton")
        self.SelTrajectorypushButton.setGeometry(QRect(255, 40, 135, 32))
        self.SelTrajectorypushButton.setStyleSheet(u"")
        self.SelTrajectorypushButton.setAutoDefault(False)
        self.TrajectorylineEdit = QLineEdit(self.groupBox)
        self.TrajectorylineEdit.setObjectName(u"TrajectorylineEdit")
        self.TrajectorylineEdit.setGeometry(QRect(400, 45, 607, 21))
        self.TrajectorylineEdit.setStyleSheet(u"")
        self.TrajectoryTypecomboBox = QComboBox(self.groupBox)
        self.TrajectoryTypecomboBox.addItem("")
        self.TrajectoryTypecomboBox.addItem("")
        self.TrajectoryTypecomboBox.setObjectName(u"TrajectoryTypecomboBox")
        self.TrajectoryTypecomboBox.setGeometry(QRect(140, 42, 110, 30))
        self.TrajectoryTypecomboBox.setStyleSheet(u"")
        self.CoregCTcomboBox = QComboBox(self.groupBox)
        self.CoregCTcomboBox.addItem("")
        self.CoregCTcomboBox.addItem("")
        self.CoregCTcomboBox.setObjectName(u"CoregCTcomboBox")
        self.CoregCTcomboBox.setEnabled(False)
        self.CoregCTcomboBox.setGeometry(QRect(205, 129, 97, 30))
        self.CoregCTcomboBox.setStyleSheet(u"")
        self.SimbNIBSTypecomboBox = QComboBox(self.groupBox)
        self.SimbNIBSTypecomboBox.addItem("")
        self.SimbNIBSTypecomboBox.addItem("")
        self.SimbNIBSTypecomboBox.setObjectName(u"SimbNIBSTypecomboBox")
        self.SimbNIBSTypecomboBox.setGeometry(QRect(140, 67, 110, 30))
        self.SimbNIBSTypecomboBox.setStyleSheet(u"")
        self.groupBox_2 = QGroupBox(Dialog)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(8, 243, 1010, 92))
        self.groupBox_2.setStyleSheet(u"")
        self.TransducerTypecomboBox = QComboBox(self.groupBox_2)
        self.TransducerTypecomboBox.addItem("")
        self.TransducerTypecomboBox.addItem("")
        self.TransducerTypecomboBox.addItem("")
        self.TransducerTypecomboBox.setObjectName(u"TransducerTypecomboBox")
        self.TransducerTypecomboBox.setGeometry(QRect(90, 34, 116, 30))
        self.TransducerTypecomboBox.setStyleSheet(u"")
        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(15, 40, 71, 16))
        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(293, 37, 151, 20))
        self.ComputingEnginecomboBox = QComboBox(self.groupBox_2)
        self.ComputingEnginecomboBox.setObjectName(u"ComputingEnginecomboBox")
        self.ComputingEnginecomboBox.setGeometry(QRect(422, 34, 275, 30))
        self.ComputingEnginecomboBox.setStyleSheet(u"")

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.ContinuepushButton.setText(QCoreApplication.translate("Dialog", u"CONTINUE", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"Imaging input", None))
        self.CoregCTlabel.setText(QCoreApplication.translate("Dialog", u"Correg.?", None))
        self.SelT1WpushButton.setText(QCoreApplication.translate("Dialog", u"Select T1W ...", None))
        self.SelCTpushButton.setText(QCoreApplication.translate("Dialog", u"Select", None))
        self.SimbNIBSlineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.SelTProfilepushButton.setText(QCoreApplication.translate("Dialog", u"Select Thermal\n"
"profile ...", None))
        self.ThermalProfilelineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.CTlineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.T1WlineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.CTTypecomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"NO", None))
        self.CTTypecomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"real CT", None))
        self.CTTypecomboBox.setItemText(2, QCoreApplication.translate("Dialog", u"ZTE", None))

        self.SelSimbNIBSpushButton.setText(QCoreApplication.translate("Dialog", u"Select SimbNIBS ...", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"Use CT?", None))
        self.SelTrajectorypushButton.setText(QCoreApplication.translate("Dialog", u"Select Trajectory ...", None))
        self.TrajectorylineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.TrajectoryTypecomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"Brainsight", None))
        self.TrajectoryTypecomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"Slicer", None))

        self.CoregCTcomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"NO", None))
        self.CoregCTcomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"CT to MRI", None))

        self.SimbNIBSTypecomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"charm", None))
        self.SimbNIBSTypecomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"headreco", None))

        self.groupBox_2.setTitle(QCoreApplication.translate("Dialog", u"Transducer and Computing engine", None))
        self.TransducerTypecomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"Single", None))
        self.TransducerTypecomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"CTX_500", None))
        self.TransducerTypecomboBox.setItemText(2, QCoreApplication.translate("Dialog", u"H317", None))

        self.label_2.setText(QCoreApplication.translate("Dialog", u"Transducer", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"Computing backend", None))
    # retranslateUi

