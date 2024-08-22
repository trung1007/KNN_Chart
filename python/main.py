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

data = json.loads(sys.argv[1])

message = data["message"]



data_to_pass_back = make_predictions(['- Em Bán Sim Vina Giống 6- 9 Số AC: 0815.496.000 - Giá Bán: 800,000 ☎️ ☎️  Liên Hệ Mua Sim : 0911929999',message], 'KNN')

print(data_to_pass_back)

# Nhận dữ liệu từ Node.js
# data = json.loads(sys.argv[1])
# # Xử lý dữ liệu
# processed_data = {
#   "message": f"Received: {data['message']}. Processed by Python!",
# }

# # In kết quả ra stdout dưới dạng JSON
# print(json.dumps(processed_data))

