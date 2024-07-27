from EvaluationMetrics import evaluate_metrics
import numpy as np
import json
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
import streamlit as st
import os, warnings
warnings.filterwarnings('ignore')
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.01, "max_new_tokens":1024},
)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

prompt_template = """
Given the following Myntra product catalog, 
generate a helpful response for the user based on these. 
While answering the question please adhere to the following guidelines:
1. Response should be completely based on the given Myntra product catalog and its answer don't add any extra knowledge or make any assumptions. 
2. Response should be clear and precise.
3. If the provided Question is not related to Myntra product catalog just respond that the given question is out of your knowledge base.
4. No pre-amble and post-amble is required, just answer the question.
FAQ and its answer:
{context}

User Question:{question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 1, 'score_threshold': 0.5}), 
    chain_type_kwargs={"prompt": custom_prompt})


def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Remove whitespace from the top of the page and sidebar
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )

# st.header("MyntraBot: Your Shopping Assistant 👗👠", divider='grey')
st.markdown("""
    <h3 style='text-align: left; color: black; padding-top: 35px; border-bottom: 3px solid purple;'>
        MyntraBot: Your Shopping Assistant👗👠
    </h3>""", unsafe_allow_html=True)

side_bar_message = """
Hi! 👋 I'm here to help you with your fashion choices. What would you like to know or explore within Myntra?
\nHere are some areas you might be interested in:
1. **Fashion Trends** 👕👖
2. **Personal Styling Advice** 👢🧢
3. **Seasonal Wardrobe Selections** 🌞
4. **Accessory Recommendations** 💍

Feel free to ask me anything about Myntra Shopping!
"""

with st.sidebar:
    # st.title(':blue_heart: Myntra Shopping Assistant')
    st.title(':purple_heart: MyntraBot')
    st.markdown(side_bar_message)

initial_message = """
    Hi there! I am here to enhance your shopping experience with Myntra. To get started, here are some questions you might ask me:\n
     🎀What are the top fashion trends this summer?\n
     🎀Can you suggest an outfit for a summer wedding?\n
     🎀What are some must-have accessories for winter season?\n
     🎀What type of shoes should I wear with a cocktail dress?\n
     🎀What's the best look for a professional photo shoot for women?
 
    Feel free to ask me anything to make your shopping easier and more stylish!"""


#    What are the latest sunglasses collection on Myntra?\n
#     How can I style a white shirt for a casual day out?\n
#     Show me the top collections available on Myntra right now?\n


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": 
                                  initial_message}]
    
# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, Let me check..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

# Evaluation section
st.sidebar.header("Evaluate MyntraBot RAG Pipeline")

if st.sidebar.button('Evaluate'):
    # product_data = json.dumps([
        # {
        #     "ProductID": "10012339",
        #     "ProductName": "Lakme Absolute Matte Melt Liquid Lip Color Rhythmic Red 20",
        #     "ProductBrand": "Lakme",
        #     "Gender": "Women",
        #     "Price": 400,
        #     "NumImages": 4,
        #     "Description": "Lakme Absolute Matte Melt Liquid Lip ColorShade name: Rhythmic Red 20",
        #     "PrimaryColor": "Red"
        # },
        # {
        #     "ProductID": "10012365",
        #     "ProductName": "Lakme Absolute Matte Melt Liquid Lip Colour 19 Red Vibe",
        #     "ProductBrand": "Lakme",
        #     "Gender": "Women",
        #     "Price": 400,
        #     "NumImages": 4,
        #     "Description": "Lakme Absolute Matte Melt Liquid Lip ColorShade name: 19 Red Vibe",
        #     "PrimaryColor": "Red"
        # }
    #     {
    #         "ProductID": "10001209",
    #         "ProductName": "Bubblegummers Boys Purple Printed Sports Sandals",
    #         "ProductBrand": "Bubblegummers",
    #         "Gender": "Boys",
    #         "Price": 699,
    #         "NumImages": 7,
    #         "Description": "A pair of purple sports sandalsSynthetic upper with velcro closureCushioned footbedPatterned rubber outsoleWarranty: 2 monthsWarranty provided by brand/manufacturer",
    #         "PrimaryColor": "Purple"
    #     }
    # ])
    
    # queries = ["What are the red lip color options available in the Lakme brand?"]
    # ground_truths = [
    #     {
    #         # 'contexts': [product_data], 
    #         'contexts':['''
    #                     {"ProductID": "10012339",
    #                     "ProductName": "Lakme Absolute Matte Melt Liquid Lip Color Rhythmic Red 20",
    #                     "ProductBrand": "Lakme",
    #                     "Gender": "Women",
    #                     "Price": 400,
    #                     "NumImages": 4,
    #                     "Description": "Lakme Absolute Matte Melt Liquid Lip ColorShade name: Rhythmic Red 20",
    #                     "PrimaryColor": "Red"
    #                    },
    #                    {
    #                     "ProductID": "10012365",
    #                     "ProductName": "Lakme Absolute Matte Melt Liquid Lip Colour 19 Red Vibe",
    #                     "ProductBrand": "Lakme",
    #                     "Gender": "Women",
    #                     "Price": 400,
    #                     "NumImages": 4,
    #                     "Description": "Lakme Absolute Matte Melt Liquid Lip ColorShade name: 19 Red Vibe",
    #                     "PrimaryColor": "Red"
    #                    }     
    #                    '''],
            
    #         'answer': "We offer two exquisite red lip color options from Lakme in our catalog: 1. Lakme Absolute Matte Melt Liquid Lip Color Rhythmic Red 20: This lip color provides a stunning matte finish in a vibrant Rhythmic Red shade, perfect for any occasion, priced at INR 400. 2. Lakme Absolute Matte Melt Liquid Lip Colour 19 Red Vibe: Another great choice for a matte finish, this lip color comes in the Red Vibe shade, offering a distinctive and bold look, also priced at INR 400."
    #     }
    # ]
    # queries = ["I want purple sandals for boys"]
    # ground_truths = [
    #     {
    #         # 'contexts':[product_data], 
    #         'contexts':["ProductID: 10001209, ProductName: Bubblegummers Boys Purple Printed Sports Sandals, ProductBrand: Bubblegummers, Gender: Boys, Price:699, Description: A pair of purple sports sandalsSynthetic upper with velcro closureCushioned footbedPatterned rubber outsoleWarranty: 2 monthsWarranty provided by brand/manufacturer, PrimaryColor:Purple "  ], 
    #         # 'answer': "Bubblegummers Boys Purple Printed Sports Sandals (ProductID: 10001209) with a price of INR 699 is available in purple color for boys. It has a cushioned footbed, patterned rubber outsole, and a warranty of 2 months provided by the brand/manufacturer."
    #         'answer': "We offer the Bubblegummers Boys Purple Printed Sports Sandals. These sandals are perfect for boys and come in a vibrant purple color. They feature a synthetic upper with velcro closure, a cushioned footbed, and a patterned rubber outsole. The sandals are priced at INR 699 and come with a 2-month warranty provided by the brand/manufacturer."
    #     }
    # ]

    queries = ["I want a sweatshirt for boy in yellow color"]
    ground_truths = [
        {
            # 'contexts':[product_data], 
            'contexts':["ProductID: 1000894, ProductName: U.S. Polo Assn. Kids Boys Yellow Hooded Sweatshirt, ProductBrand: U.S. Polo Assn. Kids, Gender: Boys, Price:899, Description: Yellow sweatshirt,Â has an attached hood with drawstring fastening,Â a full zip closure, longÂ sleeves with applique detail on one sleeve, split kangaroo pocket, an embroidered applique on the backÂ and an inner fleece lining, PrimaryColor: Yellow "  ], 
            # 'answer': "Bubblegummers Boys Purple Printed Sports Sandals (ProductID: 10001209) with a price of INR 699 is available in purple color for boys. It has a cushioned footbed, patterned rubber outsole, and a warranty of 2 months provided by the brand/manufacturer."
            'answer': '''Sure, I can help with that. Myntra offers a U.S. Polo Assn. Kids Boys Yellow Hooded Sweatshirt with the following features:
                        ProductID: 1000894
                        ProductName: U.S. Polo Assn. Kids Boys Yellow Hooded Sweatshirt
                        ProductBrand: U.S. Polo Assn. Kids
                        Gender: Boys
                        Price (INR): 899
                        Description: Yellow sweatshirt, has an attached hood with drawstring fastening, a full zip closure, long sleeves with applique detail on one sleeve, split kangaroo pocket, an embroidered applique on the back and an inner fleece lining
                        PrimaryColor: Yellow
                        This sweatshirt might be a great fit for what you're looking for.'''
        }
    ]
    	
    # queries = ["What is Python"]
    # ground_truths = [
    #     {
    #         'contexts': ["not available, out of the knowledge base, question is not related to the Myntra product catalog."], 
    #         'answer': "The provided question is out of the knowledge base as it is not related to the Myntra product catalog."
    #     }
    # ]

    metrics = evaluate_metrics(queries, ground_truths, get_response, vector_store)
    st.sidebar.subheader("Evaluation Metrics")
    for metric, values in metrics.items():
        st.sidebar.write(f"{metric}: {np.mean(values)}")
