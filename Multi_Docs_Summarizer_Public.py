# Imports the OpenAI API and Streamlit libraries.
import streamlit as st
import os
import json
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
import pandas as pd
from trubrics.integrations.streamlit import FeedbackCollector

# Create Progress Bar
class ProgressBarHandler(BaseCallbackHandler):
    current_counter = 0
    total_counter = 1
    progress_bar = None
    def __init__(
        self,
        total_counter=1
    ) -> None:
        self.total_counter = total_counter
        self.progress_text = st.empty()
        self.progress_bar = st.progress(0)
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        self.current_counter += 1
        progress_percentage = int((self.current_counter * 1.0 / (self.total_counter + 1)) * 100)  # show the progress bar meter moving
        self.progress_bar.progress(progress_percentage)
        self.progress_text.text(f"Total Chunks: {str(self.total_counter+1)}"   #show the number of chunks
                                f"\nProgress: {progress_percentage}% ({str(self.current_counter)} / {str(self.total_counter+1)})")  # show the % progress
  
        
        
def main():

    openai_token = os.environ.get("OPENAI_TOKEN", "")           
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = st.secrets.openai_version
    os.environ["OPENAI_API_BASE"] = st.secrets.openai_endpoint
    os.environ["OPENAI_API_KEY"] = st.secrets.openai_token

    st.header("""
    Notes:
    - This application is a BETA version
    - Accuracy: Due to ongoing development and the nature of the AI language model, the results may generate inaccurate or misleading information
    - Accountability: All output m
    ust be fact-checked, proof-read, and adapted as appropriate by officers for their work
    - Feedback: If you have any suggestion to improve this application, please provide them in the text box below """)
    agree = st.checkbox("I understand that I am responsible for the use of this tool as a productivity tool and that the app creator is not liable for the credibility of the results genereated.")
#     """Returns `True` if the user had the correct password."""
    if agree:

        # Add a title
        st.title("Document Summarizer")
        st.caption("AI-powered application that condense document texts into more concise and coherent summaries")

        max_summary_size = st.number_input('Max summary words(Indicative)', value=200, step=10, min_value=50)

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
                document = loader.load()
                llm = AzureChatOpenAI(temperature=0, 
                    verbose=True, 
                    deployment_name="gpt-4"
                )

                template = """Summarise the following document close to {max_summary_size} words, and capturing the main key points of the document.Ignore all footnote and references: file: {text}"""
                prompt = PromptTemplate(template = template, input_variables = ['text', 'max_summary_size'])
                chain = load_summarize_chain(llm, chain_type="refine", refine_prompt = prompt)
                splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=5000, chunk_overlap=20)
                split_documents = splitter.split_documents(document)

                handler = ProgressBarHandler(total_counter=len(split_documents))
                with st.spinner("Please wait, summarisation in process..."):
                    summarize_text = chain.run({"input_documents": split_documents, "max_summary_size": max_summary_size}, callbacks=[handler])
                    summarize_text_clean = summarize_text.replace("\n\n", " ")

                current_tries = 1
                # Ensure text is less than word limit and not too far off from its limit
                while len(re.findall(r'\w+', summarize_text_clean)) > max_summary_size and current_tries <= 5:
                    with st.spinner('Please wait ahwile longer while the model optimise the summarised result...'):
                        split_texts = splitter.split_text(summarize_text_clean)
                        split_documents = [Document(page_content=t) for t in split_texts]
                        summarize_text_clean = chain({"input_documents": split_documents, "sentence_limit": sentence_limit}, callbacks=[handler])['output_text']
                        current_tries += 1

                st.write(pdf.name, "(", len(summarize_text_clean.split()), "words )")
                st.write(summarize_text_clean)
                dict_ = {pdf.name: summarize_text_clean}
                output = {}
                output.update(dict_)
    
        # Require this to ensure proper formatting when save over as txt
            output = '\n\n'.join(['%s: \n%s' % (key, value) for (key, value) in output.items()])
            st.download_button("Download", data = output, file_name = "download.txt", mime="txt/csv")

            collector = FeedbackCollector(
                component_name = 'Multiple Documents Summariser'
                email=st.secrets['feedback_user'], password=st.secrets['feedback_pass'])
    
            collector.st_feedback(
                                  feedback_type="faces", model = 'document summarizer',
                                  open_feedback_label="Provide additional feedback") 


if __name__ == "__main__":
    main()
