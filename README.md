# LearnAI

LearnAI is a Streamlit-based application that helps users learn efficiently with artificial intelligence. It provides text summarization, keyword extraction, question generation, and doubt solving functionalities.
![image](https://github.com/user-attachments/assets/d0bbd5f5-c6c4-46f3-ace3-25384d3a1af8)
![image](https://github.com/user-attachments/assets/6a5cf390-c2c4-40bb-a42a-3432c1e1b78d)

## Features

- **Text Summarization**: Summarizes the input text to key points.
- **Keyword Extraction**: Extracts important keywords from the text.
- **Question Generation**: Generates questions based on the summarized text.
- **Doubt Solver**: Answers user-submitted doubts using a pre-trained BERT model.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Akkki28/learnai.git
    cd learnai
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
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
