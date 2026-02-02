import streamlit as st
import os
import tempfile
import time
import shutil
import gc

st.set_page_config(page_title="Industrial Extraction Engine", layout="wide")

with st.spinner("Loading dependencies..."):
  from langchain_unstructured import UnstructuredLoader
  from langchain_huggingface import HuggingFaceEmbeddings
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain_openai import ChatOpenAI
  from langchain_chroma import Chroma
  from langchain_community.vectorstores.utils import filter_complex_metadata
  from langchain_core.runnables import RunnablePassthrough
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import JsonOutputParser
  from pydantic import BaseModel, Field
  import openai


class MotorSpec(BaseModel):
  model_number: str = Field(
    description="The alphanumeric model number. Aliases: 'catalog number', 'part number', 'cat no'. Look for unique identifiers in table rows."
  )
  voltage: str = Field(
    description="Rated voltage. Look for 'volts' or 'V'. If the row uses a designator (like 'S' or 'X'), inherit the voltage from the section header (e.g., '230/460V')."
  )
  horsepower: str = Field(
    description="Rated power (HP). Aliases: 'Hp', 'Horsepower', 'Output'."
  )
  rpm: str = Field(
    description="Rotational speed. If a range is given (e.g. 0-1800), provide the full range string. Aliases: 'Speed', 'RPM'."
  )
  frame_size: str = Field(
    description="The NEMA or IEC frame size (e.g., 56C, 256T). Usually found in a 'Frame' column."
  )
  enclosure: str = Field(
    description="Enclosure type (e.g., TEFC, ODP, IP55). If not in the row, check the page header or section title."
  )


@st.cache_resource
def get_embedding_model():
  return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


DB_PATH = "./chroma_db_v2"


def process_documents(uploaded_files):
  status = st.empty()
  embedding_model = get_embedding_model()

  # Force clear internal file tracking
  st.session_state.ingested_files = []
  new_documents = []
  for file in uploaded_files:
    status.info(f"Processing {file.name}...")
    with tempfile.NamedTemporaryFile(
      delete=False, suffix=f".{file.name.split('.')[-1]}"
    ) as tmp_file:
      tmp_file.write(file.getvalue())
      tmp_path = tmp_file.name

    try:
      loader = UnstructuredLoader(tmp_path, mode="elements", strategy="fast")
      raw_docs = loader.load()

      if not raw_docs:
        st.error(f"❌ {file.name} appeared empty to the loader.")

      for doc in raw_docs:
        doc.metadata["source"] = file.name
      new_documents.extend(raw_docs)
    finally:
      os.remove(tmp_path)

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
  splits = text_splitter.split_documents(new_documents)
  splits = filter_complex_metadata(splits)

  vectorstore = Chroma.from_documents(
    documents=splits, embedding=embedding_model, persist_directory=DB_PATH
  )
  status.success(
    f"✅ Database Built with {len(st.session_state.ingested_files)} files."
  )
  time.sleep(1)
  st.rerun()


with st.sidebar:
  st.title("🛡️ Admin Control")

  if "OPENAI_API_KEY" in st.session_state and st.session_state.OPENAI_API_KEY:
    st.success("✅ OpenAI Connection Active")
    if st.button("Change Key"):
      st.session_state.OPENAI_API_KEY = ""
      st.rerun()
  else:
    api_key_input = st.text_input(
      "OpenAI API Key", type="password", help="Required for Data Extraction"
    )
    if api_key_input:
      st.session_state.OPENAI_API_KEY = api_key_input
      os.environ["OPENAI_API_KEY"] = api_key_input
      st.rerun()

  st.markdown("---")
  st.subheader("Data Management")
  uploaded_files = st.file_uploader(
    "Upload Industrial PDFs", accept_multiple_files=True
  )

  if st.button("🚀 Build New Index") and uploaded_files:
    if not st.session_state.get("OPENAI_API_KEY"):
      st.error("⚠️ Please enter an OpenAI API Key first!")
    else:
      process_documents(uploaded_files)

  if st.button("🗑️ Force Wipe Everything"):
    gc.collect()
    if os.path.exists(DB_PATH):
      try:
        shutil.rmtree(DB_PATH)
      except:
        st.error("Windows lock active. Delete 'chroma_db_X' folder manually.")

    st.session_state.ingested_files = []
    st.cache_resource.clear()
    st.success("System Reset Complete.")
    time.sleep(1)
    st.rerun()

st.title("⚙️ Industrial Spec-to-JSON Engine")

# Check if DB exists
db_ready = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0

if not db_ready:
  st.warning(
    "👈 System Offline. Please upload PDFs and click 'Build New Index' in the sidebar."
  )
else:
  st.success(
    f"✅ System Online. Index includes: {', '.join(st.session_state.get('ingested_files', ['Existing Index']))}"
  )

if user_query := st.chat_input("Ex: 'Extract specs for model EFM2515T'"):
  if not db_ready:
    st.error("No data found. Upload and Index first.")
  else:
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      try:
        embedding_model = get_embedding_model()
        db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        parser = JsonOutputParser(pydantic_object=MotorSpec)

        template = """
                You are a Senior Technical Data Specialist utilizing GPT-5's advanced reasoning.
                
                STRUCTURAL HINT (Baldor/Siemens Table Layout):
                Rows usually follow: [HP] [RPM] [Frame] [Catalog Number] [Price] [Efficiency] [Voltage]
                If they don't match exactly, adapt using spatial reasoning.

                YOUR TASK:
                1. Locate the alphanumeric Catalog Number: "{question}".
                2. Use your spatial reasoning to map the surrounding numbers to the MotorSpec schema.
                3. Note: The number 93, 94.1, etc., next to the voltage is the Efficiency percentage.
                4. INHERITANCE: If a field like 'Voltage' or 'Enclosure' is not in the row, 
                   it is 100% guaranteed to be in the section header or page title above. 
                   Scan the provided context for headers like "230/460 volts" or "Open drip proof".

                Context: 
                {context}

                User Request: {question}

                {format_instructions}
                """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
          {
            "context": db.as_retriever(search_kwargs={"k": 20})
            | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
            "format_instructions": lambda x: parser.get_format_instructions(),
          }
          | prompt
          | ChatOpenAI(model="gpt-5", temperature=0)
          | parser
        )

        with st.spinner("Processing..."):
          context_docs = db.as_retriever(search_kwargs={"k": 20}).invoke(user_query)
          combined_context = "\n\n".join(d.page_content for d in context_docs)

          # with st.expander("🔍 RAG Context"):
          #   if not combined_context.strip():
          #     st.warning("Search returned no text.")
          #   else:
          #     st.text(combined_context[:3000])

          # Execution
          result = chain.invoke(user_query)

        st.subheader("Validated Extraction")
        st.json(result)

      except Exception as e:
        st.error(f"Extraction failed: {e}")
