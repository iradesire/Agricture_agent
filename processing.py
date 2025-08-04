# #main processing point for the agricultural agent

# #load in the paths
# import os 
# import sys
# llmodel=os.path.dirname(os.path.abspath(__file__))
# folder=os.path.dirname(llmodel)
# sys.path.append(folder)

# #import the necessary libraries

# from langchain_core.vectorstores import InMemoryVectorStore
# from TOOL_GEN.Tool import detect_maize_disease, analyze_image
# from PROMPT_GEN.prompt import prompt_gen
# from EMBEDDING.embedding import embeddings
# from langchain_openai import ChatOpenAI
# from langchain.chains import LLMChain




# from langchain_community.document_loaders import PyPDFLoader

# print("Current working directory:", os.getcwd())

# # load the modelai
# api_key = os.getenv("api key")
# print = api_key 
# modelai =  ChatOpenAI(
#     api_key= api_key,
#     base_url= 'https://open.bigmodel.cn/api/paas/v4',
#     model='glm-4',
#  )


# #uploading a pdf file
# def loading_pdf(file_path):
#     loader = PyPDFLoader(file_path)
#     docs = loader.load_and_split()
#     return docs


# embedding = embeddings()

# embedding_vector_store = InMemoryVectorStore(embedding=embedding)

# # Upload and embed document
# def add_pdf_to_vectorstore(pdf_path):
#     documents = loading_pdf(pdf_path)
#     embedding_vector_store.add_documents(documents)
#     return f"Added {len(documents)} PDF chunks to the vector store."


# ## To detect disease from an image
# tools= [detect_maize_disease, analyze_image]
# def process_pdf_and_ask(pdf_path, question, agent_scratchpad):
#     add_pdf_to_vectorstore(pdf_path)
#     context_docs = embedding_vector_store.similarity_search(question)
#     context = "\n".join(doc.page_content for doc in context_docs)
#     tools_description = (
#         "- detect_maize_disease: Detects maize disease from an image.\n"
#         "- analyze_image: Analyzes the maize leaf image and provides disease name and treatment recommendation."
#     )
#     prompt = prompt_gen(question, context, tools_description, agent_scratchpad)
    
#     # Create LLM chain with the model and prompt
#     llm_chain = LLMChain(llm=modelai, prompt=prompt)
    
#     # Run the chain to get the answer
#     answer = llm_chain.invoke(
#         {"question": question, 
#          "context": context, "tools": tools_description,
#          "agent_scratchpad": agent_scratchpad})
    
#     return answer

# def process_question_only(question):
#     return f"You said: {question}"


# #Gradio Userinterface
# import gradio as gr

# def user_interface():
#     chat_ui = gr.Interface(
#         fn=process_pdf_and_ask,
#         inputs=gr.TextArea(label="What can I help with?"),
#         outputs=gr.TextArea(),
#         description="AI is happy to help you with your agricultural needs. Please upload a PDF file containing relevant information and ask your question.",
#         title="AGRICTURAL EXTENSION AGENT",
#     )
    
#     upload_ui = gr.Interface(
#         fn=add_pdf_to_vectorstore,
#         inputs=gr.File(
#         type="filepath",
#         file_count="single",
#         label="Upload your file here"
#     ),
#     outputs=gr.Textbox(label="Extracted Text")
#     )

#     output_interface = gr.TabbedInterface(
#     [chat_ui, upload_ui],
#     tab_names=["Chat", "Upload"]
#     )

#     output_interface.launch(debug=True)

# if __name__ =="__main__":
#     user_interface()
    


# Get paths
import os
import sys

llmodel = os.path.dirname(os.path.abspath(__file__))
folder = os.path.dirname(llmodel)
sys.path.append(folder)

import gradio as gr
from langchain_core.vectorstores import InMemoryVectorStore
from TOOL_GEN.Tool import detect_maize_disease, analyze_image
from PROMPT_GEN.prompt import prompt_gen
from EMBEDDING.embedding import embeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate


# Load AI model
api_key = os.getenv("api key")

modelai = ChatOpenAI(
    api_key=api_key,
    base_url='https://open.bigmodel.cn/api/paas/v4',
    model='glm-4',
)

# Load PDF and embed
def loading_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    return docs

embedding = embeddings()
embedding_vector_store = InMemoryVectorStore(embedding=embedding)

def add_pdf_to_vectorstore(pdf_path):
    documents = loading_pdf(pdf_path)
    embedding_vector_store.add_documents(documents)
    return f"Added {len(documents)} PDF chunks to the vector store."

# Image and question handler
def handle_image_and_question(image_path, question):
    result = ""
    if image_path:
        result += f"{analyze_image(image_path)}\n"
        result += f"{detect_maize_disease(image_path)}\n"
    result += f" Question received: {question}"
    return result

# Full handler (PDF + image + question)
def full_handler(pdf_path, image_path, question):
    pdf_result = add_pdf_to_vectorstore(pdf_path)
    image_result = handle_image_and_question(image_path, question)
    
    # Prepare prompt with tools and context
    context_docs = embedding_vector_store.similarity_search(question)
    context = "\n".join(doc.page_content for doc in context_docs)
    tools_description = (
        "- detect_maize_disease: Detects maize disease from an image.\n"
        "- analyze_image: Analyzes the maize leaf image and provides disease name and treatment recommendation."
    )
    prompt = prompt_gen(question, context, tools_description ,agent_scratchpad="")
    prompt = PromptTemplate.from_template(prompt_gen(question, context, tools_description, agent_scratchpad=""))

    
    llm_chain = LLMChain(llm=modelai, prompt=prompt)
    answer = llm_chain.invoke({
        "question": question,
        "context": context,
        "tools": tools_description,
        "agent_scratchpad": ""
    })

    return f"{pdf_result}\n{image_result}\n\n AI Response:\n{answer}"

# Gradio interface
def user_interface():
    interface = gr.Interface(
        fn=full_handler,
        inputs=[
            gr.File(type="filepath", label="Upload PDF"),
            gr.Image(type="filepath", label="Upload maize leaf image"),
            gr.Textbox(label=" Ask your question")
        ],
        outputs=gr.TextArea(label="Output"),
        description="Upload a PDF and/or maize leaf image, and ask a question.",
        title="Agricultural Extension Agent"
    )
    interface.launch(debug=True)

if __name__ == "__main__":
    user_interface()