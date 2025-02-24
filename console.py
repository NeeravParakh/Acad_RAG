from langchain.embeddings import SentenceTransformerEmbeddings
import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import fitz
import chromadb.utils.embedding_functions as embedding_functions
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import getpass
from tqdm.autonotebook import tqdm, trange
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from groq import Groq
from pdf2image import convert_from_path
import PIL.Image
from google import genai
import time

MAKE_CHROMA = True
FILE_PATH = r'files'
CONTEXT = ""
ANSWER_L = "L" #L or S 
QUERY = ""
PROMPT = f"""You are a highly skilled academic researcher with extensive experience in synthesizing information from scholarly articles and research papers. Your expertise lies in extracting relevant insights, summarizing complex arguments, and providing clear, concise answers to specific queries based on the provided context.
Your task is to analyze the given excerpts from scholarly articles and respond to the query accordingly.
Please ensure that your response is directly related to the query, highlighting key points from the context and providing a well-reasoned answer. Keep in mind the importance of academic rigor and clarity in your response.
Do not answer the query if there is no logical answer in the context provided and do not answer vaguely.
You will first be given context then the query.Strictly answer from the context.Give answer in very much details."""
IMAGE_PROMPT = "Extract each and every content that is written in the image, be it text,images or tables or graphs, and write it (if it is not graph  or image) in a same order as it is written in the image. Do not change anything.Keep in mind that what you write should make sense , the given image can have either normal layout or academic paper layout or any layout so check that thoroughly and write the all the content their is strictly"
client_g = genai.Client(api_key="AIzaSyDHatFdaiSOx-ii3wUsZIavWADKE3LQ3uM")
MODEL_G = "gemini-2.0-flash"
MODEL = "deepseek-r1-distill-llama-70b"
client = Groq(api_key = "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")
EMBEDDING_MODEL = r"D:\Projects\model\all-MiniLM-L6-v2"
EMBEDDING_TOKENIZER = r"D:\Projects\tokenizer\all-MiniLM-L6-v2"
PERSIST_DIRECTORY = r"chroma_db_1"