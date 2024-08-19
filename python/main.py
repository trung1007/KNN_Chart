import sys
import re
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QPushButton, QDialogButtonBox, QLabel, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit, QFileDialog
from PyQt6.QtGui import QPalette, QColor, QIcon, QPixmap, QPainter, QPolygonF
from PyQt6.QtCore import QSize, Qt, QPointF
import pandas as pd
from pathlib import Path
import KNN, naive_bayes, logistic_regression, SVM, ANN
from typing import Union
import json
import ast

def make_predictions(list_inputs, algorithm):
	vectorizer_file = 'loaded_models\\tfidf_vectorizer.pkl'
	if algorithm == 'KNN':
		model_file = 'loaded_models\\tfidf_KNN.pkl'
		return KNN.make_predictions(list_inputs=list_inputs, vectorizer_file=vectorizer_file, model_file=model_file)
	elif algorithm == 'Naive-Bayes':
		model_file = 'loaded_models\\tfidf_naive_bayes.pkl'
		return naive_bayes.make_predictions(list_inputs=list_inputs, vectorizer_file=vectorizer_file, model_file=model_file)
	elif algorithm == 'Logistic Regression':
		model_file = 'loaded_models\\tfidf_logistic_regression.pkl'
		return logistic_regression.make_predictions(list_inputs=list_inputs, vectorizer_file=vectorizer_file, model_file=model_file)
	elif algorithm == 'SVM':
		model_file = 'loaded_models\\tfidf_SVM.pkl'
		return SVM.make_predictions(list_inputs=list_inputs, vectorizer_file=vectorizer_file, model_file=model_file)
	elif algorithm == 'ANN':
		model_file = 'loaded_models\\tfidf_ANN.h5'
		return ANN.make_predictions(list_inputs=list_inputs, vectorizer_file=vectorizer_file, model_file=model_file)

# data_to_pass_back = make_predictions(['- Em Bán Sim Vina Giống 6- 9 Số AC: 0815.496.000 - Giá Bán: 800,000 ☎️ ☎️  Liên Hệ Mua Sim : 0911929999','aaaaaa'], 'KNN')

# Nhận dữ liệu từ Node.js
data = json.loads(sys.argv[1])
# Xử lý dữ liệu
processed_data = {
  "message": f"Received: {data['message']}. Processed by Python!",
}

# In kết quả ra stdout dưới dạng JSON
print(json.dumps(processed_data))
# def alert(message:str, parent=None):
# 	dlg = QMessageBox(parent)
# 	dlg.setWindowTitle("Message from the Application")
# 	dlg.setText(message)
# 	dlg.exec()

# class TextDialog(QDialog):
# 	def __init__(self, parent):
# 		super().__init__(parent)
# 		self.setWindowTitle("Text Dialog Box")
# 		self.resize(600, 600)

# 		layout = QVBoxLayout(self)

# 		scroll_area = QScrollArea(self)
# 		scroll_area.setWidgetResizable(True)

# 		text_edit = QTextEdit()
# 		text_edit.setAcceptRichText(False)
		
# 		scroll_area.setWidget(text_edit)
# 		layout.addWidget(scroll_area)
# 		ok_button = QPushButton("OK")
# 		ok_button.clicked.connect(self.get_text)
# 		layout.addWidget(ok_button)
# 		self.text_edit = text_edit

# 	def get_text(self):
# 		text = self.text_edit.toPlainText()
# 		self.parent().addTextInput(text)  # You can replace this with your desired action upon clicking OK
# 		self.accept()

# class RightArrowButton(QPushButton):
# 	def __init__(self, coords:tuple, parent=None):
# 		super().__init__(parent)
# 		self.setFixedSize(100, 40)  # Set the size of the button
# 		self.setGeometry(coords[0], coords[1], 100, 40)
# 		icon = QIcon()
# 		pixmap = QPixmap(20, 20)
# 		pixmap.fill(Qt.GlobalColor.transparent)
        
# 		painter = QPainter(pixmap)
# 		painter.setRenderHint(QPainter.RenderHint.Antialiasing)

# 		# Draw right arrow on the pixmap
# 		points = [QPointF(5, 5), QPointF(15, 10), QPointF(5, 15)]
# 		painter.drawPolygon(QPolygonF(points))

# 		icon.addPixmap(pixmap, QIcon.Mode.Normal, QIcon.State.Off)
# 		self.setIcon(icon)
# 		self.setIconSize(QSize(20, 20))
# 		del painter
# 		del pixmap

# class AboutDlg(QDialog):
# 	def __init__(self, parent=None):
# 		super().__init__(parent)
# 		self.setWindowTitle("Additional Information")
# 		self.setFixedSize(1100,600)
# 		manual = QLabel(self)
# 		about_text = f'This is the Spam Filter App, created by group 11 as the product of the Introduction to Artificial Intelligence class'
# 		manual.setText(about_text)
# 		manual.show()

# class MainWindow(QMainWindow):
# 	def __init__(self):
# 		super(MainWindow, self).__init__()
# 		self.setWindowTitle("Spam Filter")
# 		self.setFixedSize(1000,800)
# 		self.setStyleSheet('background-color: #AEB3BD;')
# 		self.algorithm = 'KNN'

#         # ------------------------------ Spam App start ----------------------------------
# 		self.list_inputs = []
# 		self.list_outputs = []
# 		self.predicted = False # This variable keeps track of whether the predictions have been made
		
# 		self.InputLabel = QLabel("User Text", self)
# 		self.InputLabel.setStyleSheet('''
# 		font-size: 26px;
# 		''')
# 		self.InputLabel.setGeometry(158, 50, 350, 50)
# 		self.InputList = QWidget(self)
# 		self.InputList.setGeometry(50, 100, 350, 550)
# 		self.InputList_layout = QVBoxLayout() # The main layout of the input list, vertical
# 		self.InputList_scroll_area = QScrollArea() # The area for the list of the current text inputs
# 		self.InputList_area_widget = QWidget()
# 		self.InputList_area_widget.setStyleSheet('''
# 		background-color: #FFFFFF;
# 		color: #000000;
# 		''')
# 		self.InputList_area_layout = QVBoxLayout(self.InputList_area_widget)
# 		self.InputList_area_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
# 		self.InputList_scroll_area.setWidget(self.InputList_area_widget)
# 		self.InputList_scroll_area.setWidgetResizable(True)
# 		self.InputList_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
# 		self.InputList_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
# 		self.InputList_layout.addWidget(self.InputList_scroll_area)
		
# 		self.InputList_panel_layout = QHBoxLayout()
# 		self.InputList_panel_add = QPushButton()
# 		self.InputList_panel_add.setText("Add text")
# 		self.InputList_panel_add.clicked.connect(self.CreateTextDialog)
# 		self.InputList_panel_addFile = QPushButton()
# 		self.InputList_panel_addFile.setText("Read text from file")
# 		self.InputList_panel_addFile.clicked.connect(self.InputFromFile)
# 		self.InputList_panel_clear = QPushButton()
# 		self.InputList_panel_clear.setText("Clear")
# 		self.InputList_panel_clear.clicked.connect(self.RemoveInputList)
# 		self.InputList_panel_layout.addWidget(self.InputList_panel_add)
# 		self.InputList_panel_layout.addWidget(self.InputList_panel_addFile)
# 		self.InputList_panel_layout.addWidget(self.InputList_panel_clear)
		
# 		self.InputList_layout.addLayout(self.InputList_panel_layout)
# 		self.InputList.setLayout(self.InputList_layout)
# 		self.InputList.show()
		

# 		self.OutputLabel = QLabel("Predictions", self)
# 		self.OutputLabel.setGeometry(711, 50, 350, 50)
# 		self.OutputLabel.setStyleSheet('''
# 		font-size: 26px;
# 		''')
# 		self.OutputList = QWidget(self)
# 		self.OutputList.setGeometry(600, 100, 350, 550)
# 		self.OutputList_layout = QVBoxLayout() # The main layout of the Output list, vertical
# 		self.OutputList_scroll_area = QScrollArea() # The area for the list of the current predictions
# 		self.OutputList_area_widget = QWidget()
# 		self.OutputList_area_widget.setStyleSheet('''
# 		background-color: #FFFFFF;
# 		color: #000000;
# 		''')
# 		self.OutputList_area_layout = QVBoxLayout(self.OutputList_area_widget)
# 		self.OutputList_area_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
# 		self.OutputList_scroll_area.setWidget(self.OutputList_area_widget)
# 		self.OutputList_scroll_area.setWidgetResizable(True)
# 		self.OutputList_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
# 		self.OutputList_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
# 		self.OutputList_layout.addWidget(self.OutputList_scroll_area)
		
# 		self.OutputList_panel_layout = QHBoxLayout()
# 		self.OutputList_panel_export = QPushButton()
# 		self.OutputList_panel_export.setText("Export")
# 		self.OutputList_panel_export.clicked.connect(self.OutputExport)
# 		self.OutputList_panel_clear = QPushButton()
# 		self.OutputList_panel_clear.setText("Clear")
# 		self.OutputList_panel_clear.clicked.connect(self.RemoveOutputList)
# 		self.OutputList_panel_layout.addWidget(self.OutputList_panel_export)
# 		self.OutputList_panel_layout.addWidget(self.OutputList_panel_clear)
		
# 		self.OutputList_layout.addLayout(self.OutputList_panel_layout)
# 		self.OutputList.setLayout(self.OutputList_layout)
# 		self.OutputList.show()

# 		self.PredictLabel = QLabel("PREDICT", self)
# 		self.PredictLabel.setGeometry(467, 280, 100, 50)
# 		self.PredictButton = RightArrowButton((450,330), self)
# 		self.PredictButton.show()
# 		self.PredictButton.clicked.connect(self.predict)

# 		self.AlgoLabel = QLabel("Choose an algorithm", self)
# 		self.AlgoLabel.setGeometry(415, 170, 180, 50)
# 		self.AlgoComboBox = QComboBox(self)
# 		self.AlgoComboBox.setGeometry(410, 220, 180, 50)
# 		self.AlgoComboBox.addItem('KNN')
# 		self.AlgoComboBox.addItem('Naive-Bayes')
# 		self.AlgoComboBox.addItem('Logistic Regression')
# 		self.AlgoComboBox.addItem('SVM')
# 		self.AlgoComboBox.addItem('ANN')
# 		self.AlgoComboBox.currentTextChanged.connect(self.on_algorithm_changed)
#         # ------------------------------ Spam App end ------------------------------------

# 	def InputFromFile(self):
# 		dialog = QFileDialog(self)
# 		dialog.setDirectory(str(Path('./')))
# 		dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
# 		dialog.setNameFilter("Text files (*.txt)")
# 		dialog.setViewMode(QFileDialog.ViewMode.List)
# 		if dialog.exec():
# 			filenames = dialog.selectedFiles()
# 			if filenames:
# 				for filename in filenames:
# 					content = open(filename, 'r',encoding='utf-8').read()
# 					self.addTextInput(content)
		
# 		return
	
# 	def CreateTextDialog(self):
# 		text_dialog = TextDialog(self)
# 		text_dialog.show()
# 		return
	
# 	def addTextInput(self, text:str):
# 		all_lines = re.sub(r'([\n]+)', "\n", text).split('\n')
# 		for line in all_lines:
# 			if line:
# 				d = QLabel(line)
# 				self.InputList_area_layout.addWidget(d)
# 				self.InputList_scroll_area.update()
# 				self.list_inputs.append(line)
# 		print(self.list_inputs)

# 	def OutputExport(self):
# 		df = pd.DataFrame()
# 		try:
# 			assert self.predicted and len(self.list_inputs) == len(self.list_outputs) and len(self.list_inputs) > 0
# 		except AssertionError:
# 			alert('Please provide some text data for the program and ensure that you have cliked the PREDICT button!', self)
# 			return
# 		df['text'] = self.list_inputs
# 		df['spam'] = self.list_outputs
# 		options = QFileDialog.Option(QFileDialog.Option.DontUseNativeDialog)

# 		file_name, _ = QFileDialog.getSaveFileName(
# 			None,
# 			"Save CSV File",
# 			"",
# 			"CSV Files (*.csv)",
#         	options=options
# 		)

# 		if file_name:  # Check if a file name was selected
# 			if not file_name.lower().endswith('.csv'):  # Ensure that the file has a .csv extension
# 				file_name += '.csv'
# 			# print(f"Selected file: {file_name}")
# 			df.to_csv(file_name, index=False) # Save the file in csv format with two columns named "text" and "spam"

# 	def RemoveInputList(self):
# 		self.list_inputs = []
# 		index = self.InputList_area_layout.count()-1
# 		while (index>-1):
# 			myWidget = self.InputList_area_layout.itemAt(index).widget()
# 			myWidget.setParent(None)
# 			index -= 1
# 		self.RemoveOutputList()

# 	def RemoveOutputList(self):
# 		self.list_outputs = []
# 		index = self.OutputList_area_layout.count()-1
# 		while (index>-1):
# 			myWidget = self.OutputList_area_layout.itemAt(index).widget()
# 			myWidget.setParent(None)
# 			index -= 1
# 		self.predicted = False

# 	def predict(self):
# 		self.predicted = True
		
# 		# This debug program will make random predictions for testing
# 		self.list_outputs = make_predictions(self.list_inputs, self.algorithm)
# 		print(self.list_outputs)

# 		# This is result of all message by int
# 		with open("result.txt", "w") as function: 
# 			function.write( " ".join([str(int(x)) for x in self.list_outputs]))
		
# 		# This is result of spam message
# 		with open("spamMessage.txt", "w", encoding="utf-8") as function:
# 			spamMessage = []
# 			for i, message in enumerate(self.list_outputs) :
# 				if(int(message) == 1):
# 					spamMessage.append(self.list_inputs[i])
# 			function.write("\n".join(spamMessage))
			
# 		# For the real program, you should replace the above line with self.list_outputs = make_predictions(self.list_inputs, self.algorithm) where make_predictions is a function which takes a list of strings and return a list with the same number of elements which only contains 1 (for spam) and 0 (non-spam)
# 		assert len(self.list_inputs) == len(self.list_outputs)
# 		for i in range(len(self.list_outputs)):
# 			label = self.list_outputs[i]
# 			d = QLabel(f'[{str(label)}] {self.list_inputs[i]}')
# 			if label:
# 				d.setStyleSheet("""
# 				background-color: #FA1818;
# 				color: #000000;
# 				font-family: Titillium;
# 				font-size: 18px;
# 				""")
# 			else:
# 				d.setStyleSheet("""
# 				background-color: #7AFFED;
# 				color: #000000;
# 				font-family: Titillium;
# 				font-size: 18px;
# 				""")
# 			self.OutputList_area_layout.addWidget(d)
# 			self.OutputList_scroll_area.update()

# 		print(self.list_outputs)
# 	def on_algorithm_changed(self):
# 		self.algorithm = self.AlgoComboBox.currentText()
# 		print(self.algorithm)


# app = QApplication(sys.argv)

# window = MainWindow()
# window.show()

# try:
# 	app.exec()
# except Exception as e:
# 	app.quit()
