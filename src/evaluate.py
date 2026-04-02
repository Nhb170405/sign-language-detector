import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

y_true = np.array(['Cảm ơn', 'Xin lỗi', 'Xin chào', 'Cảm ơn', 'Cảm ơn', 'Xin chào'])
y_pred = np.array(['Cảm ơn', 'Xin chào', 'Xin chào', 'Cảm ơn', 'Xin lỗi', 'Xin chào'])
classes = np.unique(y_true)

cm = confusion_matrix(y_true, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Confusion Matrix', ha='center', fontsize=13)

plt.show()

print(classification_report(y_true, y_pred, target_names=classes))