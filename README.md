# LearnAI

LearnAI is a Streamlit-based application that helps users learn efficiently with artificial intelligence. It provides text summarization, keyword extraction, question generation, and doubt solving functionalities.

## Features

- **Text Summarization**: Summarizes the input text to key points.
- **Keyword Extraction**: Extracts important keywords from the text.
- **Question Generation**: Generates questions based on the summarized text.
- **Doubt Solver**: Answers user-submitted doubts using a pre-trained BERT model.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/learnai.git
    cd learnai
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download NLTK data:
    ```sh
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('brown'); nltk.download('wordnet')"
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Enter the material you want to study in the text area and click "Analyze".

4. View the summarized key points, generated questions, and use the doubt solver to get answers to your questions.

## File Structure

- [app.py](http://_vscodecontentref_/0): Main application file containing the Streamlit app.
- [functions.py](http://_vscodecontentref_/1): Contains various helper functions for loading models, text processing, summarization, keyword extraction, question generation, and doubt solving.
- [requirements.txt](http://_vscodecontentref_/2): Lists the required Python packages.

## License

This project is licensed under the MIT License.
