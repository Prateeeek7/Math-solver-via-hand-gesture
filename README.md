
# ✋ Math Solver via Hand Gesture

A computer vision-based project that reads hand gestures through your webcam and interprets them as mathematical expressions. It then solves the equation in real-time and displays the result.

## 🚀 Features

- 🖐️ Recognizes hand gestures using MediaPipe
- ➕ Performs arithmetic operations: `+`, `-`, `*`, `/`, `=`
- 🧠 Real-time gesture tracking using OpenCV
- 🔢 Supports one-handed and two-handed gesture recognition
- 🎯 High detection confidence (set to 0.7)
- 🧮 Evaluates expressions like `1 + 2`, `3 * 4`, etc.

## 📦 Technologies Used

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## 🧑‍💻 How It Works

- One thumb up = **"+"**
- One thumb and one finger = **"-"**
- One thumb and two fingers = **"*"**
- One thumb and three fingers = **"/"**
- Fist = **"="**
- Detects digits from **1 to 9**

## 📂 Project Structure

```
hand-gesture-math-solver/
├── main.py                 # Main logic for gesture detection and math solving
├── README.md               # Project documentation
└── requirements.txt        # Required Python packages
```

## 🔧 Setup & Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Prateeeek7/Math-solver-via-hand-gesture.git
   cd Math-solver-via-hand-gesture
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   ```bash
   python main.py
   ```

## 📈 Future Improvements

- Add support for more gestures (e.g., `0`, `C` for clear)
- Add voice feedback
- Export results as PDF or speech

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

## 📜 License

[MIT](LICENSE)
