## Downloading the data
## Index the data the data with its embeddings to a DocumentStore (preprocessing, cleaning and splitting)
import gdown ## Download files from Gdrive
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
# from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

## Download business data files to local data
data_gdrive = "https://drive.google.com/drive/folders/1n9yqq5Gl_HWfND5bTlrCwAOycMDt5EMj"
output_dir = "data"
gdown.download_folder(data_gdrive, quiet=True, output=output_dir)

## Indexing and loading the file data into Document store
## Future feature: Using Prod grade document store like Chroma
document_store = InMemoryDocumentStore()

## Future feature: Adding more file types
file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
pdf_converter = PyPDFToDocument()
document_joiner = DocumentJoiner()

## Cleaning and chunking: Clean for whitespace. Split words into chunks of 150 words, with bit of overlap to avoid missing context
document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)

## Embedding the chunks: Create embeddings using nvidia embedding (https://build.ngc.nvidia.com/explore/) 
document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
## Future feature: Using a good quality embedder like Nvidia Embed QA
# document_embedder = NvidiaDocumentEmbedder(
#     model="NV-Embed-QA",
#     api_url="https://ai.api.nvidia.com/v1/retrieval/nvidia",
#     api_key=Secret.from_token("NVIDIA_API_KEY"),
# )
document_embedder.warm_up()

## Write to document store database: Write Documents to the DocumentStore
document_writer = DocumentWriter(document_store)

## Add all elements to one RAG pipeline
preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

## Connect all the pipeline elements
preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
preprocessing_pipeline.connect("text_file_converter", "document_joiner")
preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
preprocessing_pipeline.connect("markdown_converter", "document_joiner")
preprocessing_pipeline.connect("document_joiner", "document_cleaner")
preprocessing_pipeline.connect("document_cleaner", "document_splitter")
preprocessing_pipeline.connect("document_splitter", "document_embedder")
preprocessing_pipeline.connect("document_embedder", "document_writer")

preprocessing_pipeline.run({"file_type_router": {"sources": list(Path(output_dir).glob("**/*"))}})
