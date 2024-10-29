# Serendipity AI - Mental Health Concern Classification

**Serendipity AI** is a comprehensive AI-powered journaling and mental health analysis tool designed to aid users in self-care, mental health tracking, and therapy support. Built with natural language processing (NLP) capabilities, Serendipity AI provides users and therapists with structured insights, emotion tracking, and personalized journaling approaches. This project was developed for **Megathon 2024** in collaboration with Mindpeers.

## Motivation and Objective

Serendipity AI aims to make mental health management accessible and user-friendly through AI-driven journaling and analytics. By combining text analysis, concern classification, and sentiment tracking, it offers users a structured approach to understand their emotional landscape over time and facilitates communication with therapists.

## Features

1. **Journaling Interface**:
   - **Minimal Design**: Easy-to-use interface to log journal entries with quick navigation and privacy settings.
   - **Emotion Tracking**: Displays the top five emotions detected and their polarity.
   - **Customizable Privacy**: Share analytics or specific journal entries with a therapist as desired.

2. **Analytics Dashboard**:
   - **Emotion Analytics**: Tracks emotional patterns over time (weekly, monthly) with time series data.
   - **Consistency Insights**: Analyzes journaling consistency, mood swings, and provides visual summaries.
   - **Custom Notes**: Users can add reflections on their analytic insights.

3. **Two Approaches to Journaling**:
   - **Free-form Journaling**: Users can freely express thoughts without prompts.
   - **Guided Journaling**: An AI-driven assistant offers insightful prompts, helping users express emotions methodically.

4. **AI Chatbot for Mental Health Assistance**:
   - Powered by the Mistral Nemo model, the chatbot provides personalized journaling assistance and conversation summaries.
   - Helps users explore deeper emotions with insightful questions and allows for context-specific follow-ups.

## Underlying Pipeline

The core Serendipity AI pipeline comprises:
- **Key-Phrase Extraction**: Identifies symptoms and behaviors from journal entries, powered by Clinical BERT.
- **Polarity Modeling**: Uses the VADER Sentiment Analysis tool for reliable, rule-based sentiment scoring.
- **Concern Classification**: Classifies concerns using a fine-tuned RoBERTa_Large model for improved accuracy.
- **Time Series Analysis**: Visualizes emotional trends, supporting both user insights and therapist monitoring.

## Dataset

Serendipity AI is trained on a dataset derived from mental health-focused subreddits, specifically designed to handle long-form text entries.

## Installation

### Prerequisites
- Python 3.x
- Install dependencies from `requirements.txt`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/godofwarnings/Serendipity-ai.git
   cd Serendipity-ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Fine-Tuning Model**:
   ```bash
   python finetune.py --data [path_to_data] --output [output_model_path]
   ```
2. **Start Console Application**:
   ```bash
   python consoleApp.py
   ```
   Follow prompts to interact with the journaling assistant and explore analytics.

## Future Work

Potential improvements include:
- Integration with health monitoring devices for a holistic view (e.g., combining journaling with physical metrics).
- Multi-modal inputs (e.g., voice, photos).
- Advanced analytics, including sleep and chronic disorder detection.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Clinical BERT](https://arxiv.org/abs/1904.03323)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [Mindpeers - Megathon 2024](https://megathon.mindpeers.com)
