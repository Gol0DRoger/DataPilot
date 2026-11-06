import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.schema import ETLAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool


#------------------------------------ AGENT TOOLS ------------------------------------#


@tool
def extract_load_tool(url:str, output_folder:str, format:str) -> str:
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()
    return etl_tools.extract_load(url, output_folder, format)


@tool
def transform_load_tool(input_file_path:str,output_folder:str,output_format:str, user_question:str) -> str:
    """
    This tool transforms the data from the specified file (CSV, Excel, JSON or Parquet) and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv or xlsx).
        user_question (str): The transformation to perform, in plain English.
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()

    try:
        data_context = etl_tools.transform_load_context(input_file_path)
    except Exception as e:
        return f"Could not read the input file: {e}"

    llm = pick_llm("low")  # simple Pandas code generation - the cheaper model is enough

    # The LLM only writes the transformation. Loading and saving the file is done by ETLTools.
    prompt = f"""
            You are a Python Data Analyst who uses Pandas. Write only Python code - no explanation, no comments, no markdown.
            A Pandas DataFrame named `df` is already loaded from the file {input_file_path}. `pd` and `np` are available.
            Do not import anything and do not read or write any files.
            Transform `df` as requested and assign the final DataFrame to a variable named `result_df`.

            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {data_context}\n
        """

    response = llm.invoke(prompt).content

    # Cleaning: remove optional Markdown fences and stray pandas/numpy imports
    pandas_code = response.strip()
    pandas_code = re.sub(r"^```(?:python)?\s*", "", pandas_code, flags=re.IGNORECASE)
    pandas_code = re.sub(r"\s*```$", "", pandas_code).strip()
    pandas_code = re.sub(r"^\s*(?:import|from)\s+(?:pandas|numpy)\b.*$", "", pandas_code, flags=re.MULTILINE).strip()

    # Run the code on the loaded data and save the result
    return etl_tools.transform_load(input_file_path, output_folder, output_format, pandas_code)


# Toolkit 
tools = [extract_load_tool, transform_load_tool]

tools_by_name = {t.name: t for t in tools}

llm_bind = pick_llm("medium").bind_tools(tools)  # same model as "high" with lower reasoning effort (saves tokens)


# ---------------------------------------- AGENT GRAPH ---------------------------------------- #

def llm_node(state:ETLAgentSchema):

    system_prompt = SystemMessage(content="""
            You are a Python Data Analyst who has access to tools that can extract and load, 
            transform and load data. You will be provided with a user's question 
            and you would need to perform the right ETL operations as per the user's question. 
            Use the file paths, folders and formats exactly as given by the user.
            If the operation is performed then inform the user and end the coversation.
    """)

    final_answer = llm_bind.invoke([system_prompt] + state.messages)

    # Return only the new message - the 'add' reducer appends it to the history
    return {"messages": [final_answer]}


def tool_node(state:ETLAgentSchema):
    """
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    """

    tools_results = []

    tool_calls = state.messages[-1].tool_calls

    for tool_call in tool_calls:

        try:
            observation = tools_by_name[tool_call['name']].invoke(tool_call['args'])
        except Exception as e:
            observation = f"Tool failed: {e}"

        tools_results.append(ToolMessage(content=str(observation), tool_call_id = tool_call['id']))

    return {"messages": tools_results}


# Nodes & Edges
etl_analyst_graph = StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")

def is_tool_call(state:ETLAgentSchema):
    tool_calls = state.messages[-1].tool_calls

    if tool_calls:
        return "tool_node"
    else:
        return "end"

etl_analyst_graph.add_conditional_edges(
    "llm_node",is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END
    }
)

etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()

if __name__ == "__main__":
    # Compile the Graph
    

    # Optional
    from IPython.display import display, Image
    img = Image(etl_analyst.get_graph().draw_mermaid_png())
    with open("etl_analyst_graph.png", "wb") as f:
        f.write(img.data)

    '''response = etl_analyst.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")]}
    )'''

    response = etl_analyst.invoke(
          {"messages":[HumanMessage(content=f"""
             I want to transform the data stored in the 'C:\\Users\\risha\\Downloads\\Data_agent\\data\\extract\\extracted_data.csv' file 
             and save the transformed data in the 'C:\\Users\\risha\\Downloads\\Data_agent\\data\\transform' folder in the csv format.
             The transformation should filter the data to show bulbasaur pokemon only.
 """)]}
     )    

    print(response)