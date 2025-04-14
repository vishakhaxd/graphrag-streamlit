from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
import json
import spacy
from collections import Counter
from pathlib import Path
from wasabi import msg
from spacy_llm.util import assemble
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


directory_path = '/Users/koro/Documents/nlp-books'
loaders = [PyPDFLoader(os.path.join(directory_path, f)) for f in os.listdir(directory_path) if f.endswith('.pdf')]

docs = []
for loader in loaders:
    docs.extend(loader.load())

paper_data = [
    Document(
        page_content=doc.page_content,
        metadata={
            "source": doc.metadata.get('source', '').removeprefix('/Users/koro/Documents/nlp-books'),
            "page": doc.metadata.get('page', None),
            "title": doc.metadata.get('title', 'Unknown Title'),
            "author": doc.metadata.get('author', 'Unknown Author'),
            "date": doc.metadata.get('mod_date', 'Unknown Date'),
            "keywords": doc.metadata.get('keywords', []),
        }
    )
    for doc in docs
]


all_data = paper_data


# Initialize the text splitter
rtext_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
llm = ChatOpenAI(temperature=0, model_name="gpt-4")


# Define the map prompt template
map_template = """The following is a set of documents
{all_data}
Based on this list of docs, please perform concise summaries while extracting essential relationships for relationships analysis later on. Please answer questions about the material
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)

map_chain = LLMChain(llm=llm, prompt=map_prompt)
all_data = paper_data
reduce_template = """The following is set of summaries:
{all_data}
Take these and distill it into concise summaries of the chapters while containing important relationships. Example: CNNs use kernels, which showcases not only the relationship between CNN and kernel.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,
    document_variable_name="all_data"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(

    combine_documents_chain=combine_documents_chain,

    collapse_documents_chain=combine_documents_chain,

    token_max=4000,
)

map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,

    reduce_documents_chain=reduce_documents_chain,

    document_variable_name="all_data",

    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(all_data)

# Run the MapReduce Chain
summarization_results = map_reduce_chain.run(split_docs)
print(summarization_results)
with open('summary.txt', 'w') as file:
    file.write(str(summarization_results))


# Entitites

def split_document_sent(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]  # referencial


# spacy-llm relationship extraction
def process_text(nlp, text, verbose=False):
    doc = nlp(text)
    if verbose:
        msg.text(f"Text: {doc.text}")
        msg.text(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
        msg.text("Relations:")
        for r in doc._.rel:
            msg.text(f"  - {doc.ents[r.dep]} [{r.relation}] {doc.ents[r.dest]}")
    return doc


def run_pipeline(config_path, examples_path=None, verbose=False):
    if not os.getenv("OPENAI_API_KEY"):
        msg.fail("OPENAI_API_KEY env variable was not found. Set it and try again.", exits=1)

    nlp = assemble(config_path, overrides={} if examples_path is None else {"paths.examples": str(examples_path)})

    # Initialize counters and storage
    processed_data = []
    entity_counts = Counter()
    relation_counts = Counter()

    # Load your articles and news data here
    # all_data = news_articles_data + documents

    sents = split_document_sent(summarization_results)
    for sent in sents:
        doc = process_text(nlp, sent, verbose)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        relations = [(doc.ents[r.dep].text, r.relation, doc.ents[r.dest].text) for r in doc._.rel]

        # Store processed data
        processed_data.append({'text': doc.text, 'entities': entities, 'relations': relations})

        # Update counters
        entity_counts.update([ent[1] for ent in entities])
        relation_counts.update([rel[1] for rel in relations])

    # Export to JSON
    with open('processed_data.json', 'w') as f:
        json.dump(processed_data, f)

    # Display summary
    msg.text(f"Entity counts: {entity_counts}")
    msg.text(f"Relation counts: {relation_counts}")


config_path = Path("zeroshot.cfg")
examples_path = None
verbose = True

# Run the pipeline
file = run_pipeline(config_path, None, verbose)
from neo4j import GraphDatabase

uri = "neo4j+s://01454400.databases.neo4j.io"
auth = ("neo4j", "Ws-Gbf5TvjmXgju4igjvcYWHeIt9j8IFIc8A0Rq0r2o")

driver = GraphDatabase.driver(uri, auth=auth)


def query_graph(query):
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]


# Example query to get all nodes and relationships
query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10;"
data = query_graph(query)
print(data)
