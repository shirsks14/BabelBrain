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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QGroupBox, QLabel, QPushButton,
    QSizePolicy, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(665, 246)
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(6, 3, 651, 206))
        self.groupBox.setAutoFillBackground(False)
        self.groupBox.setStyleSheet(u"")
        self.ElastixOptimizercomboBox = QComboBox(self.groupBox)
        self.ElastixOptimizercomboBox.addItem("")
        self.ElastixOptimizercomboBox.addItem("")
        self.ElastixOptimizercomboBox.addItem("")
        self.ElastixOptimizercomboBox.setObjectName(u"ElastixOptimizercomboBox")
        self.ElastixOptimizercomboBox.setGeometry(QRect(326, 31, 281, 30))
        self.ElastixOptimizercomboBox.setLayoutDirection(Qt.LeftToRight)
        self.ElastixOptimizercomboBox.setStyleSheet(u"")
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(134, 37, 201, 16))
        self.grpManualFOV = QGroupBox(self.groupBox)
        self.grpManualFOV.setObjectName(u"grpManualFOV")
        self.grpManualFOV.setEnabled(False)
        self.grpManualFOV.setGeometry(QRect(231, 122, 206, 72))
        self.FOVDiameterSpinBox = QDoubleSpinBox(self.grpManualFOV)
        self.FOVDiameterSpinBox.setObjectName(u"FOVDiameterSpinBox")
        self.FOVDiameterSpinBox.setGeometry(QRect(105, 7, 90, 22))
        self.FOVDiameterSpinBox.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.FOVDiameterSpinBox.setDecimals(1)
        self.FOVDiameterSpinBox.setMinimum(10.000000000000000)
        self.FOVDiameterSpinBox.setMaximum(400.000000000000000)
        self.FOVDiameterSpinBox.setSingleStep(0.100000000000000)
        self.FOVDiameterSpinBox.setValue(200.000000000000000)
        self.DiameterLabel = QLabel(self.grpManualFOV)
        self.DiameterLabel.setObjectName(u"DiameterLabel")
        self.DiameterLabel.setGeometry(QRect(9, 40, 82, 20))
        self.FOVLengthSpinBox = QDoubleSpinBox(self.grpManualFOV)
        self.FOVLengthSpinBox.setObjectName(u"FOVLengthSpinBox")
        self.FOVLengthSpinBox.setGeometry(QRect(105, 40, 90, 22))
        self.FOVLengthSpinBox.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.FOVLengthSpinBox.setDecimals(1)
        self.FOVLengthSpinBox.setMinimum(200.000000000000000)
        self.FOVLengthSpinBox.setMaximum(600.000000000000000)
        self.FOVLengthSpinBox.setSingleStep(0.100000000000000)
        self.FOVLengthSpinBox.setValue(400.000000000000000)
        self.FocalLengthLabel = QLabel(self.grpManualFOV)
        self.FocalLengthLabel.setObjectName(u"FocalLengthLabel")
        self.FocalLengthLabel.setGeometry(QRect(9, 7, 94, 20))
        self.ManualFOVcheckBox = QCheckBox(self.groupBox)
        self.ManualFOVcheckBox.setObjectName(u"ManualFOVcheckBox")
        self.ManualFOVcheckBox.setGeometry(QRect(201, 98, 150, 20))
        self.ManualFOVcheckBox.setLayoutDirection(Qt.RightToLeft)
        self.ForceBlendercheckBox = QCheckBox(self.groupBox)
        self.ForceBlendercheckBox.setObjectName(u"ForceBlendercheckBox")
        self.ForceBlendercheckBox.setGeometry(QRect(1, 70, 349, 20))
        self.ForceBlendercheckBox.setLayoutDirection(Qt.RightToLeft)
        self.CancelpushButton = QPushButton(Dialog)
        self.CancelpushButton.setObjectName(u"CancelpushButton")
        self.CancelpushButton.setGeometry(QRect(584, 214, 74, 32))
        self.ContinuepushButton = QPushButton(Dialog)
        self.ContinuepushButton.setObjectName(u"ContinuepushButton")
        self.ContinuepushButton.setGeometry(QRect(230, 216, 136, 32))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"Domain Generation Parameters", None))
        self.ElastixOptimizercomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"AdaptiveStochasticGradientDescent", None))
        self.ElastixOptimizercomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"FiniteDifferenceGradientDescent", None))
        self.ElastixOptimizercomboBox.setItemText(2, QCoreApplication.translate("Dialog", u"QuasiNewtonLBFGS", None))

        self.label_5.setText(QCoreApplication.translate("Dialog", u"Elastix co-registration Optimizer", None))
        self.grpManualFOV.setTitle("")
        self.DiameterLabel.setText(QCoreApplication.translate("Dialog", u"Length (mm)", None))
        self.FocalLengthLabel.setText(QCoreApplication.translate("Dialog", u"Diameter (mm)", None))
        self.ManualFOVcheckBox.setText(QCoreApplication.translate("Dialog", u"Manual Subvolume  ", None))
        self.ForceBlendercheckBox.setText(QCoreApplication.translate("Dialog", u"Force using Blender for Constructive Solid Geometry   ", None))
        self.CancelpushButton.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
        self.ContinuepushButton.setText(QCoreApplication.translate("Dialog", u"Ok", None))
    # retranslateUi

