# Imports the OpenAI API and Streamlit libraries.
# import pip_system_certs.wrapt_requests
import streamlit as st
import os
import langchain
from langchain.document_loaders import UnstructuredFileIOLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from typing import Dict, Any, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document
import re
import io

class ProgressBarHandler(BaseCallbackHandler):
    current_counter = 0
    total_counter = 1
    progress_bar = None
    def __init__(
        self,
        total_counter=1
    ) -> None:
        self.total_counter = total_counter
        self.progress_bar = st.progress(0, text=f"Total chunks: {str(total_counter)}")
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        self.current_counter += 1
        self.progress_bar.progress(int((self.current_counter * 1.0 / (self.total_counter + 1)) * 100))  
        
        
def main():

    openai_token = os.environ.get("OPENAI_TOKEN", "")
    openai_endpoint = "https://mti-nerve-openai-jp-east.openai.azure.com/"
           
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = openai_endpoint
    os.environ["OPENAI_API_KEY"] = "5d15498f2d0f4bbfa507c01fb859912e" 

    if check_password():
    # Set page title
#         st.set_page_config(page_title="Document Summarizer")

        # Add a title
        st.title("Document Summarizer")
        st.caption("AI-powered application that condense document texts into more concise and coherent summaries")

        max_summary_size = st.number_input('Max summary words(Indicative)', value=200, step=10, min_value=50)
        num_summaries = st.slider("Number of Summaries (Note: more summaries requested, the longer the wait time)", min_value=1, max_value=3, step=1, value=1)

        # Add a file uploader widget
        multi_pdf = st.file_uploader("Choose a file", accept_multiple_files=True)


    #     # Check if a file was uploaded
        if multi_pdf is not None and st.button("Generate"):
            output = {}
            for pdf in multi_pdf:
                file_content = pdf.read()
                file_like_object = io.BytesIO(file_content)

                # Read the contents of the file
                loader = UnstructuredFileIOLoader(file_like_object)
                split_documents = loader.load_and_split()
                llm = AzureChatOpenAI(temperature=0, 
                    verbose=True, 
                    deployment_name="gpt-35-turbo-16k"
                )

                template = """Summarise the following paper to about {word_limit} words: " + "paper: {text}"""
                prompt = PromptTemplate(template = template, input_variables = ['text', 'word_limit'])

                chain = load_summarize_chain(llm, chain_type="refine", refine_prompt = prompt)

        #         splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=8000, chunk_overlap=20)


        #         split_documents = splitter.split_documents(documents)

                handler = ProgressBarHandler(total_counter=len(split_documents))

                summarize_text = chain({"input_documents": split_documents, "word_limit":max_summary_size}, return_only_outputs = True)['output_text']
                summarize_text_clean = summarize_text.replace("\n\n", " ")

                summaries = []

        #         current_tries = 1

        #         while (len(re.findall(r'\w+', summarize_text_clean)) > max_summary_size or len(re.findall(r'\w+', summarize_text_clean)) < max_summary_size-50) and current_tries <= num_summaries:
        #             split_texts = splitter.split_text(summarize_text)
        #             split_documents = [Document(page_content=t) for t in split_texts]
        #             summarize_text = chain({"input_documents": split_documents, "word_limit":max_summary_size}, return_only_outputs =True)['output_text']
        #             summaries.append(summarize_text)
        #             current_tries += 1

                # Ensure text is less than word limit and not too far off from its limit
                if len(re.findall(r'\w+', summarize_text_clean)) < max_summary_size and len(re.findall(r'\w+', summarize_text_clean)) > max_summary_size-50:
                    summaries.append(summarize_text_clean)
                else:
                    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=5000, chunk_overlap=20)
                    split_texts = splitter.split_text(summarize_text_clean)
                    split_documents = [Document(page_content=t) for t in split_texts]
                    summarize_text_clean = chain({"input_documents": split_documents, "word_limit":max_summary_size}, return_only_outputs =True)['output_text']
                    summaries.append(summarize_text_clean)

                for i in range(num_summaries-1):
                    summarize_text_clean2 = chain({"input_documents": split_documents, "word_limit":max_summary_size}, return_only_outputs = True)['output_text']
                    while len(summarize_text_clean2.split()) == len(summaries[i].split()):
                        summarize_text_clean2 = chain({"input_documents": split_documents, "word_limit":max_summary_size}, return_only_outputs = True)['output_text']
                    summaries.append(summarize_text_clean2)


                st.write(pdf.name)
                for index, summary in enumerate(summaries):
                    index +=1
                    name = [pdf.name]
                    summarised = [summary]
                    dict_ = dict(zip(name, summarised))
                    output |= dict_
                    st.write("Summary: ", index, "(", len(summary.split()), "words )")
                    st.write(summary)


            # Require this to ensure proper formatting when save over as txt
            output = '\n\n'.join(['%s:: %s' % (key, value) for (key, value) in output.items()])



            st.download_button("Download", data = output, file_name = "download.txt", mime="text/csv")
            st.button("Regenerate")

def check_password():
    st.header("""
    Notes:
    - This application is a BETA version
    - Data Security: Please only input information classified up to Official (Closed) / Non-Sensitive
    - Accuracy: Due to ongoing development and the nature of the AI language model, the results may generate inaccurate or misleading information
    - Accountability: All output must be fact-checked, proof-read, and adapted as appropriate by officers for their work
    - Feedback: If you have any suggestion to improve this application, please email: mti-do_helpdesk@mti.gov.sg """)
    agree = st.checkbox("I acknowledge that AI language models may generate inaccurate or misleading information. I understand that this is meant as a productivity tool. I will double check and adapt generated output for appropriate use")
#     """Returns `True` if the user had the correct password."""
    if agree:
        def password_entered():
            """Checks whether a password entered by the user is correct."""


            if st.session_state["password"] == st.secrets["password"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # don't store password
            else:
                st.session_state["password_correct"] = False

        if "password_correct" not in st.session_state:
            # First run, show input for password.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            return False
        elif not st.session_state["password_correct"]:
            # Password not correct, show input + error.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            st.error("😕 Password incorrect")
            return False
        else:
            # Password correct.
            return True


if __name__ == "__main__":
    main()
