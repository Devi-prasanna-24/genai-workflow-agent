
import os
import re
from datetime import datetime

import gradio as gr
import torch
from pypdf import PdfReader
from transformers import pipeline

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1
)


def read_document(file_path):
    if not file_path:
        raise ValueError("Please upload a TXT or PDF file.")

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    elif extension == ".pdf":
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    else:
        raise ValueError("Only TXT and PDF files are supported.")

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError(
            "No readable text found. Scanned PDFs need OCR support."
        )

    return text


def make_chunks(text, words_per_chunk=450):
    words = text.split()
    return [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def create_checklist(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keywords = (
        "should", "must", "need to", "required",
        "recommend", "action", "improve", "ensure",
        "implement", "review", "develop", "protect"
    )

    tasks = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 5:
            continue

        if any(word in sentence.lower() for word in keywords):
            tasks.append(sentence)

        if len(tasks) == 7:
            break

    if not tasks:
        tasks = [
            "Review the document's main findings.",
            "Verify important facts and recommendations.",
            "Decide which follow-up actions are necessary."
        ]

    return tasks


def analyze_document(file_path, instruction):
    try:
        text = read_document(file_path)

        chunks = make_chunks(text)

        summaries = []

        for chunk in chunks:
            words = len(chunk.split())

            if words < 40:
                summaries.append(chunk)
                continue

            output = summarizer(
                chunk,
                max_length=min(120, max(40, words // 2)),
                min_length=min(30, max(10, words // 8)),
                do_sample=False,
                truncation=True
            )

            summaries.append(output[0]["summary_text"])

        summary = " ".join(summaries)
        checklist = create_checklist(text)

        instruction = (instruction or "").strip()
        if not instruction:
            instruction = "Summarize the document and identify action items."

        report = f"""# Document Analysis Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## User Instruction

{instruction}

## Document Statistics

- Word count: {len(text.split())}
- Text chunks processed: {len(chunks)}
- AI model: {MODEL_NAME}

## Summary

{summary}

## Suggested Action Checklist

""" + "\n".join(
            f"- [ ] {task}" for task in checklist
        )

        report += """

## Review Notice

This report was generated automatically. Verify important facts
and suggested actions against the original document before use.
"""

        output_path = "/content/workflow_report.md"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        return summary, "\n".join(
            f"- [ ] {task}" for task in checklist
        ), report, output_path

    except Exception as error:
        raise gr.Error(f"Could not process document: {error}")


with gr.Blocks(title="GenAI Workflow Automation Agent") as demo:
    gr.Markdown("# GenAI Workflow Automation Agent")
    gr.Markdown(
        "Upload a document to generate an AI summary, "
        "a suggested action checklist, and a report."
    )

    document = gr.File(
        label="Upload a TXT or PDF document",
        file_types=[".txt", ".pdf"],
        type="filepath"
    )

    instruction = gr.Textbox(
        label="Workflow instruction",
        placeholder="Summarize the document and list important actions...",
        value="Summarize the document, identify key findings, "
              "and create an action checklist."
    )

    run_button = gr.Button("Analyze Document", variant="primary")

    summary_output = gr.Textbox(
        label="AI Summary",
        lines=8
    )

    checklist_output = gr.Textbox(
        label="Suggested Action Checklist",
        lines=8
    )

    report_output = gr.Markdown(label="Generated Report")

    download_output = gr.File(label="Download Report")

    run_button.click(
        fn=analyze_document,
        inputs=[document, instruction],
        outputs=[
            summary_output,
            checklist_output,
            report_output,
            download_output
        ]
    )


if __name__ == "__main__":
    demo.launch(share=True)
