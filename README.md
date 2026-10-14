# Automated-Content-Extraction-System

## Unleash the Power of Learning from Math Videos!

The Math Video Summarizer is an innovative web application designed to help students, educators, and enthusiasts quickly grasp the core concepts, formulas, and examples from any math-related YouTube video. Say goodbye to endless scrubbing and re-watching – get straight to the knowledge you need!

## ✨ Features

* **Smart Summarization**: Get concise, intelligent summaries of long math lectures, allowing you to quickly understand the main topics and takeaways.
* **Formula Extraction**: Automatically identify and extract LaTeX-style math formulas and equations directly from the video's content (transcriptions/subtitles). Formulas are rendered beautifully using MathJax.
* **Clear Examples**: Pinpoint and present step-by-step solved examples, making it easier to follow along with problem-solving methodologies.
* **Video Details**: Access essential information about the video, including its title, duration, upload date, and channel.
* **User Feedback & Adaptation**: Provide feedback on summary quality, helping the system learn and adapt to generate even better summaries for you in the future.
* **Multiple Summary Styles**: Choose from various summary styles (Short, Medium, Long, Formal, Simplified) to suit your learning preference.
* **Quality Assessment**: Get an AI-driven quality score indicating how well the summary matches the original content.

## 🚀 How It Works

1.  **Paste YouTube Link**: Enter the URL of any math-focused YouTube video into the input field.
2.  **Select Summary Style**: Choose your preferred summary length and style (e.g., "Short Summary", "Formal Style").
3.  **Click "Get Summary"**: The application processes the video by:
    * Downloading available subtitles or transcribing the audio (if no subtitles are present) using `yt-dlp` and `faster-whisper`.
    * Utilizing advanced NLP models (`distilbart-cnn-12-6` and `google/flan-t5-base`) to generate a comprehensive summary.
    * Intelligently extracting mathematical formulas and examples from the text.
4.  **Explore the Output**: Review the generated summary, a list of extracted formulas, and detailed examples, all presented in an easy-to-read format.
5.  **Provide Feedback**: Let us know if the summary was helpful! Your feedback helps improve the system for everyone.

## 🛠️ Technologies Used

* **Frontend**:
    * React and Vite
    * Responsive CSS and Lucide icons
* **Backend**:
    * Flask (Python web framework)
    * `yt-dlp`: For downloading YouTube video information and subtitles/audio.
    * `faster-whisper`: For efficient audio transcription.
    * `transformers` (Hugging Face): Leveraging pre-trained models like `sshleifer/distilbart-cnn-12-6` (for general summarization) and `google/flan-t5-base` (for instruction-tuned summarization and formula extraction).
    * `webvtt`: For parsing VTT subtitle files.
    * `word2number`: For converting number words to digits in mathematical expressions.
    * `nltk` & `rouge-score`: For evaluating summarization quality (BLEU and ROUGE metrics).
    * `sqlite3`: For persistent storage of user feedback and summary quality data.

## 🚦 Installation and Setup

### Prerequisites

* Python 3.8+
* `ffmpeg` (must be installed and its `bin` directory added to your system's PATH, or the path explicitly provided in `app.py`). You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

### Steps

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/math-video-summarizer.git](https://github.com/YOUR_USERNAME/math-video-summarizer.git)
    cd math-video-summarizer
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install and build the React frontend:**

    ```bash
    npm install
    npm run build
    ```
    *(Create a `requirements.txt` file with the following content if you don't have one):*
    ```
    Flask
    yt-dlp
    whisper
    transformers
    webvtt-py
    faster-whisper
    torch # For GPU support, install appropriate version from pytorch.org
    word2number
    nltk
    rouge-score
    ```
    *Note: For `torch`, consider installing the correct version for your system (with or without CUDA) from [pytorch.org](https://pytorch.org/get-started/locally/). If you face issues with `whisper` or `faster-whisper`, refer to their respective GitHub pages for specific installation instructions.*

5.  **Run the Flask application:**

    ```bash
    python app.py
    ```

6.  **Access the application:**

    Open your web browser and go to `http://127.0.0.1:5000/`.

For frontend development with hot reload, run `npm run dev` in a second terminal while Flask is running at `http://127.0.0.1:5000`.

## 📂 Project Structure
```
maths-summarizer/
├── app.py
├── Dockerfile
├── feedback.db
├── payload.json
├── requirements.txt
├── downloads/
├── hf_models/
└── templates/
    └── index.html
```


## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the MIT License.

---
**Built with 💙 for learners everywhere.**
