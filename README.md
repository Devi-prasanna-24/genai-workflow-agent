# GenAI-Based Intelligent Workflow Automation Agent

## Project Overview

This project uses Generative AI and NLP to analyze documents,
generate summaries, identify possible action items, and create
downloadable reports.

## Features

- Upload TXT and PDF documents
- Generate AI-powered summaries
- Extract suggested action items
- Create downloadable Markdown reports
- Process long documents in smaller chunks
- Display document statistics

## Technologies

- Python
- Hugging Face Transformers
- DistilBART
- Gradio
- PyTorch
- pypdf

## Installation

Install the required libraries:

pip install -r requirements.txt

## Run the Application

python app.py

## How to Use

1. Upload a TXT or PDF document.
2. Enter a workflow instruction.
3. Click Analyze Document.
4. Review the summary and checklist.
5. Download the generated report.

## Model

sshleifer/distilbart-cnn-12-6

Model page:
https://huggingface.co/sshleifer/distilbart-cnn-12-6

## Limitations

The initial version uses a pretrained summarization model.
The workflow instruction is recorded but does not yet control
a general-purpose LLM agent.

The action checklist uses simple text rules. It may miss tasks
or identify irrelevant sentences.

Verify AI-generated output before relying on it.
