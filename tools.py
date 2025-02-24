from console import *
import streamlit as st 
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Custom embedding function subclassing Chroma's EmbeddingFunction
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL)

    def __call__(self, input: Documents) -> Embeddings:
        # Tokenize the input documents
        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Convert embeddings to list of lists
        embeddings = sentence_embeddings.tolist()

        return embeddings
    def embed_documents(self, documents: Documents) -> Embeddings:
        # Reuse the __call__ method for embedding documents
        return self.__call__(documents)
    
    def embed_query(self, query: str) -> list[float]:
        # Tokenize the query
        encoded_query = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_query)

        # Perform pooling
        query_embedding = mean_pooling(model_output, encoded_query['attention_mask'])

        # Normalize embeddings
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

        # Convert to a list
        return query_embedding.squeeze().tolist()
    
def generate_tokens(s, file_path):
    # Determine the chunk size based on the length of the input string
    # n = len(s)
    # if n <= 1000:
    #     l = 200
    # elif n <= 2500:
    #     l = 500
    # elif n <= 5000:
    #     l = 1000
    # elif n <= 7500:
    #     l = 1500
    # else:
    #     l = 2000
    i = 1
    for str in s:
    # Use RecursiveCharacterTextSplitter to split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(        
            separators=["\n\n"]
        )
        
        # Split the text and create documents
        splits = text_splitter.split_text(str)
        documents = text_splitter.create_documents(splits)

    # Add file path metadata to each document
        for doc in documents:
            doc.metadata['file_path'] = file_path
            doc.metadata['file_name'] = os.path.basename(file_path)
            doc.metadata['page'] = i
            i +=1

    # Define the persist directory and collection name
        persist_directory = PERSIST_DIRECTORY
        collection_name = "vector_store"

    # Create a Chroma vector database from the documents
    
        vectordb = Chroma.from_documents(
            documents=documents, 
            embedding=MyEmbeddingFunction(), 
            persist_directory=persist_directory, 
            collection_name=collection_name
        )
    return None

def generate_text_from_image(image, client_g, MODEL_G, IMAGE_PROMPT, max_retries=5):
    """Attempts to generate text from an image with retries on rate limit errors."""
    retry_delay = 2  # Initial delay in seconds
    for attempt in range(max_retries):
        try:
            response = client_g.models.generate_content(
                model=MODEL_G,
                contents=[IMAGE_PROMPT, image]
            )
            return response.text
        except KeyError as e:
            if "rate limit" in str(e).lower():
                st.warning(f"Rate limit hit. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                st.error(f"An error occurred: {e}")
                return None
    st.error("Max retries reached. Skipping this image.")
    return None


def process_pdf(file_path):
    doc = fitz.open(file_path)
    raw_text = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))  # Increase resolution
        image_path = f'page_{page_num}.png'
        pix.save(image_path)
        
        try:
            image = PIL.Image.open(image_path)
            text = generate_text_from_image(image)
            if text:
                raw_text.append(text)
                st.markdown(text)
        finally:
            os.remove(image_path)  # Ensure file cleanup
    
    return raw_text


def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        

        # RAW_TEXT IS IN UNICODE SO DON'T PRINT OR TRY TO ENCODE
        # IF WRITING IN TEXT FILE THEN USE ENCODING = "UTF-8"
        if file_extension == '.pdf':
            #process_pdf(file_path)
            doc = fitz.open(file_path)
            response = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))  # Increase resolution by scaling
                pix.save(f'page_{page_num}.png')  # Save with high DPI
                image = PIL.Image.open(f'page_{page_num}.png')
                retry_delay = 2  # Initial delay in seconds
                for attempt in range(5):
                    try:
                        a = client_g.models.generate_content(
                            model=MODEL_G,
                            contents=[IMAGE_PROMPT, image]
                        )
                        response.append(a.text)
                        st.markdown(a.text)
                        break
                    
                    except KeyError as e:
                        if "rate limit" in str(e).lower():
                            st.warning(f"Rate limit hit. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            st.error(f"An error occurred: {e}")
                
                os.remove(f'page_{page_num}.png')

            # raw_text = ""
            
            # for page_num in range(len(doc)):
            #     page = doc[page_num]
            #     text = page.get_text("text")
            #     raw_text += (text + "\n")
            return response
        else:
            print("Skipped")
            return ''
    except Exception as e:
        print("ERROR:>>>>>>>>>")
        print(file_path)
        print(e)
        return ''
    
def extract_text_from_folder(folder_path):
    #all_text = ''
    all_text = []
    for file_path in glob.glob(os.path.join(folder_path, '**'), recursive=True):
        st.write("Extracting ",file_path)
        if os.path.isfile(file_path):
            text = extract_text_from_file(file_path)
            print(file_path)
            if text:
                all_text.append(text)
            print(text)
            generate_tokens(text,file_path,)
            print("X")
    return all_text

def get_answer(question):
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY,collection_name="vector_store",embedding_function= MyEmbeddingFunction())
    matching_docs = vectordb.similarity_search(question,k=3)
    print(matching_docs)
    if matching_docs:
        CONTEXT = "\n".join(doc.page_content for doc in matching_docs)
        QUERY = question
        final =  PROMPT + "Query : " + QUERY + "\n\nContext : " + CONTEXT
        if ANSWER_L == "L":
            answer = client.chat.completions.create(
                    model= MODEL,
                    messages=[{"role": "user", "content": final}],
                    temperature=0.1
                )
            answer = answer.choices[0].message.content
        if ANSWER_L == "S":
            response = client_g.models.generate_content(
                    model=MODEL_G,
                    contents= final)
            answer = response.text
        return [answer,matching_docs]


    else:
        answer = "No such information present"
        return [answer]
