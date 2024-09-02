# %% [markdown]
# RAG - REACT - TOOL

# %%
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True) 

# %%
def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate an embedding model from OpenAI (smaller version for efficiency)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  

    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 

    return vector_store  # Return the created vector store

# %%
def load_embeddings_chroma(persist_directory='./chroma_db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate the same embedding model used during creation
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536) 

    # Load a Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings) 

    return vector_store  # Return the loaded vector store

# %%
# Loading the pdf document into LangChain 
#Load the documents
from langchain_community.document_loaders import DirectoryLoader
DATA_PATH = "mydata"
loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
data = loader.load()

# %%
def chunk_data(data, chunk_size, chunk_overlap): #make this 512 anc see u shall not get the result
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# %%
from langchain.vectorstores import Chroma
# Splitting the document into chunks
chunks = chunk_data(data, chunk_size=1000, chunk_overlap=50) 

# %%
print("####### FIRST 10 docs ############")
for i in range(10):
    print("\n")
    print(chunks[i])

print("####### LAST 10 docs ############")
x = chunks[::-1]
i = 0
while (i < 10):
    print("\n")
    print(x[i])
    i+= 1

print("\n\n")
print("############## # of Documents ##################")
print(len(chunks))

# %%
# Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
vector_store = create_embeddings_chroma(chunks)

# %%
def ask_and_get_answer(vector_store, q, k=5):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.invoke(q)
    return answer

# %%
# Asking questions
#question = 'Does WMi support json interfaces?'
#question = 'Who is the current prime minister of the UK?'
#question = 'what is the answer to 3 ** 3.9'
#question = 'Tell me about Napoleon Bonaparte early life'
#question = 'how to enable "Putaway To Empty Non-Replenishable Dynamic Active Locations" in Manhattan Active Supply Chain'
#question = 'how to enable "Putaway To Empty Non-Replenishable Dynamic Active Locations" in Manhattan Active® Warehouse Management?'
#question = 'Is the feature "Creation of Multiple Process Needs for an Item" in Manhattan Active Supply Chain, enabled by default?'
#question = 'Is the feature "Creation of Multiple Process Needs for an Item" in Manhattan Active® Warehouse Management, enabled by default?'
#question = 'What is the maximum weight and volume supported for transportaion order size in Manhattan Active TM?'
#question = 'what is the default maximum weight and volume supported?'
question = 'what is the default maximum weight supported in lbs?'
#question = 'Explain transportation order alerts and missing fields'

vector_store = load_embeddings_chroma()
answer = ask_and_get_answer(vector_store, question)
#print(answer)
print(answer['result'])

# %%
# We can't ask follow-up questions. There is no memory (chat history) available.
question = 'Multiply that number by 2.'
answer = ask_and_get_answer(vector_store, question)
print(answer['result'])

# %%
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain  # Import class for building conversational AI chains 
from langchain.memory import ConversationBufferMemory  # Import memory for storing conversation history

# Instantiate a ChatGPT LLM (temperature controls randomness)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)  

# Configure vector store to act as a retriever (finding similar items, returning top 5)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})  


# Create a memory buffer to track the conversation
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

crc = ConversationalRetrievalChain.from_llm(
    llm=llm,  # Link the ChatGPT LLM
    retriever=retriever,  # Link the vector store based retriever
    memory=memory,  # Link the conversation memory
    chain_type='stuff',  # Specify the chain type
    verbose=False  # Set to True to enable verbose logging for debugging
)

# %%
# create a function to ask questions
def ask_question(q, chain):
    result = chain.invoke({'question': q})
    return result

# %%
vector_store = load_embeddings_chroma()

# Asking questions
#question = 'Does WMi support json interfaces?'
#question = 'Who is the current prime minister of the UK?'
#question = 'what is the answer to 3 ** 3.9'
#question = 'Tell me about Napoleon Bonaparte early life'
#question = 'how to enable "Putaway To Empty Non-Replenishable Dynamic Active Locations" in Manhattan Active Supply Chain'
#question = 'how to enable "Putaway To Empty Non-Replenishable Dynamic Active Locations" in Manhattan Active® Warehouse Management?'
#question = 'Is the feature "Creation of Multiple Process Needs for an Item" in Manhattan Active Supply Chain, enabled by default?'
#question = 'Is the feature "Creation of Multiple Process Needs for an Item" in Manhattan Active® Warehouse Management, enabled by default?'
#question = 'What is the maximum weight and volume supported for transportaion order size in Manhattan Active TM?'
#question = 'what is the default maximum weight and volume supported?'
question = 'what is the default maximum weight supported in lbs?'
#question = 'Explain transportation order alerts and missing fields'

vector_store = load_embeddings_chroma()

result = ask_question(question, crc)
print(result['answer'])

# %%
question = 'Multiply that number by 10.'
result = ask_question(question, crc)
print(result['answer'])

# %%
question = 'Devide the result by 40.'
result = ask_question(question, crc)
print(result['answer'])

# %%
for item in result['chat_history']:
    print(item)


