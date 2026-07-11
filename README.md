# 💬 SMS Spam Detection using LSTM (Deep Learning)

An end-to-end Deep Learning and Natural Language Processing (NLP) pipeline built with **TensorFlow** to detect and filter out fraudulent or unsolicited SMS messages. Leveraging recurrent architectures, this system effectively captures sequential text context and long-range semantic dependencies, achieving an outstanding **98% classification accuracy**.

---

## 📊 Model Performance Highlights

- **Core Framework:** TensorFlow / Keras
- **Training Accuracy:** ~98% 🚀
- **Optimization:** EarlyStopping callback integration to prevent overfitting and preserve optimal weights based on validation loss tracking.

---

## 🧠 Model Architecture & Deep Learning Core

- **Text Preprocessing:** Implements tokenization, sequencing, and padding layouts to transform text data into uniform tensor formats.
- **Embedding Layer:** Maps words into high-dimensional dense vector spaces to maintain semantic relationships.
- **Bidirectional LSTM Layers:** Utilizes stacked, bidirectional LSTM cells to process sequences forward and backward, capturing deep contextual dependencies that traditional models miss.
- **Regularization & Output:** Integrates Dropout layers to mitigate overfitting, followed by a Dense layer with a Sigmoid activation function for crisp binary classification.

---

## 🛠️ Tech Stack & Dependencies

- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Natural Language Processing:** NLTK / Keras Tokenizer
- **Data Science Tools:** Python 3.x, NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn (For Loss/Accuracy curves and Confusion Matrix plots)

A high-performance SMS Spam Detection system powered by Deep Learning (TensorFlow). Features an NLP pipeline utilizing an LSTM network to capture long-term text dependencies, achieving a standout 98% classification accuracy in filtering real-time ham and spam messages.
