import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from streamlit_agraph import agraph, Node, Edge, Config
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain

import openai
import os


# Function to process the query and return a response
def process_query(query, graph):
    # Use GraphCypherQAChain to get a Cypher query and a natural language response
    try:
        result = cypher_chain(query)
        intermediate_steps = result.get("intermediate_steps", [])
        final_answer = result.get("result", "No result found.")
        generated_cypher = intermediate_steps[0].get("query", "") if intermediate_steps else None

        # Fetch graph data using the Cypher query
        nodes, edges = fetch_graph_data(direct_cypher_query=generated_cypher, graph=graph)
        return final_answer, nodes, edges
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return "Error occurred", [], []


# Function to fetch data from Neo4j
def fetch_graph_data(nodesType=None, relType=None, direct_cypher_query=None, graph=None):
    try:
        if direct_cypher_query:
            result = graph.query(direct_cypher_query)
            return process_graph_result(result)
        elif nodesType or relType:
            cypher_query = construct_cypher_query(nodesType, relType)
            result = graph.query(cypher_query)
            return process_graph_result(result)
        else:
            return [], []
    except Exception as e:
        st.error(f"Error fetching graph data: {e}")
        return [], []


# Function to construct the Cypher query based on selected filters
def construct_cypher_query(node_types, rel_types):
    if not node_types:
        node_types = ["Entity"]
    if not rel_types:
        rel_types = []

    node_clauses = [f"(p:{node_type})-[r]->(n)" for node_type in node_types]
    rel_clauses = [f"type(r)='{rel_type}'" for rel_type in rel_types]

    if rel_clauses:
        rel_match = " OR ".join(rel_clauses)
        query = f"MATCH {' OR '.join(node_clauses)} WHERE {rel_match} RETURN p, r, n"
    else:
        query = f"MATCH {' OR '.join(node_clauses)} RETURN p, r, n"
    return query


# Process graph results into nodes and edges for visualization
def process_graph_result(result):
    nodes = []
    edges = []
    node_names = set()

    for record in result:
        p_name = record["p"]["name"]
        n_name = record["n"]["name"]

        if p_name not in node_names:
            nodes.append(Node(id=p_name, label=p_name, size=5, shape="circle"))
            node_names.add(p_name)
        if n_name not in node_names:
            nodes.append(Node(id=n_name, label=n_name, size=5, shape="circle"))
            node_names.add(n_name)

        r = record["r"]
        relationship_label = r["type"]
        if "date" in r:
            relationship_label += f" ({r['date']})"
        edges.append(Edge(source=p_name, target=n_name, label=relationship_label))

    return nodes, edges


st.write("Loaded Secrets:", st.secrets)

# Initialize Neo4j and OpenAI
st.title("The OpenAI Saga")
NEO4J_URI = st.secrets['NEO4J']["URI"]
NEO4J_USERNAME = st.secrets['NEO4J']["USERNAME"]
NEO4J_PASSWORD = st.secrets['NEO4J']["PASSWORD"]

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

# Sidebar filters
st.sidebar.header("Filters")
node_types = ["Entity", "Topic","Organisation"]
relationship_types = [
    "BELONGS_TO", "IS_BETTER THAN", "DERIVED FROM", "PART_OF", "DEVELOPED_BY", "HAS_CHARACTERISTICS"
]

selected_node_types = st.sidebar.multiselect("Node Types", node_types, default=node_types)
selected_relationship_types = st.sidebar.multiselect("Relationship Types", relationship_types,
                                                     default=relationship_types)

# OpenAI API Key
openai_api_key = ""


# Query input
if prompt := st.text_input("Ask a question"):
    if not openai_api_key:
        st.error("Please enter an OpenAI API key.")
    else:
        cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4",openai_api_key=openai_api_key),
            qa_llm=ChatOpenAI(temperature=0, model_name="gpt-4",openai_api_key=openai_api_key),
            graph=graph,
            allow_dangerous_requests=True
        )
        response_structured, nodes, edges = process_query(prompt, graph)
        st.write(f"Response: {response_structured}")
        config = Config(height=600, width=800, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
        agraph(nodes=nodes, edges=edges, config=config)

