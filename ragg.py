from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama


pdf_reader = PyPDFLoader("David_Martin_Urology_Contract_Oct1_2021.pdf")
pdf_pages = pdf_reader.load()   


text_cutter = RecursiveCharacterTextSplitter(
    chunk_size=300,      
    chunk_overlap=50     
)

small_pieces = text_cutter.split_documents(pdf_pages)

all_texts = []
for piece in small_pieces:
    all_texts.append(piece.page_content)

magic_model = SentenceTransformer("all-MiniLM-L6-v2")
all_numbers = magic_model.encode(all_texts)

def find_best_text(question, how_many=3):

    question_number = magic_model.encode([question])[0]
    score_and_text = []

    for i in range(len(all_numbers)):
        text_number = all_numbers[i]

        dot = np.dot(question_number, text_number)
        size_q = np.linalg.norm(question_number)
        size_t = np.linalg.norm(text_number)

        similarity = dot / (size_q * size_t)

        score_and_text.append((similarity, all_texts[i]))

    score_and_text.sort(reverse=True)

    best_text = ""
    for i in range(how_many):
        best_text += score_and_text[i][1] + "\n\n"

    return best_text

def ask_ai(question):

    notes = find_best_text(question)

    prompt = f"""
You are a expert in AI.
Read the notes and answer.

NOTES:
{notes}

QUESTION:
{question}

ANSWER:
"""

    ai_reply = ollama.generate(
        model="gemma3:1b",
        prompt=prompt
    )

    return ai_reply["response"]

print("My RAG is Ready")

while True:
    q = input("Ask me anything from the PDF: ")

    if q.lower() == "exit":
        print("Bye ")
        break

    answer = ask_ai(q)
    print("\nAI Says:", answer)
