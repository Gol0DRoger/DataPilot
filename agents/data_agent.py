import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.schema import RouterSchema, DataAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst


llm = pick_llm("medium")  # routing is a simple classification

llm_router = llm.with_structured_output(
    RouterSchema,
    method="json_schema",
    strict=True
)


# ---------------------------- DATA AGENT GRAPH ---------------------------- #


def router_node(state: DataAgentSchema):

    message = state.messages[-1].content

    prompt = f"""
    Classify the user's request into one category:

    - etl: Extracting data from APIs/files, transforming data,
           cleaning data, or loading data.
    - sql: Querying or analyzing data stored in a database.

    Do not answer the user's request.
    Only classify it.

    User request: {message}
    """

    route_response_dict = llm_router.invoke(prompt).model_dump()
    route_response = route_response_dict['answer']

    return {"route_response": route_response}



def etl_node(state: DataAgentSchema):

    messages = state.messages

    # Add the input file / output folder info from the UI (if any) to the latest user message
    if state.etl_hint:
        messages = messages[:-1] + [HumanMessage(content=f"{messages[-1].content}\n\n{state.etl_hint}")]

    response = etl_analyst.invoke(
        {"messages": messages},
        config={"recursion_limit": 10}  # stops retry loops from burning the Groq token limit
    )

    final_message = response["messages"][-1]

    # Tool results contain the Pandas code that was executed (shown in the UI)
    details = "\n\n".join(m.content for m in response["messages"] if isinstance(m, ToolMessage))

    return {
        "messages": [final_message],
        "details": details
    }

def sql_node(state: DataAgentSchema):

    message = state.messages[-1].content

    # Give the SQL agent the recent conversation so follow-up questions make sense
    history = "\n".join(f"{m.type}: {m.content}" for m in state.messages[-5:-1])
    question = f"Conversation so far:\n{history}\n\nCurrent question: {message}" if history else message

    input_schema = {
        "messages": [],
        "user_question": question,
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": "",
        "db_url": state.db_url
    }

    response = sql_analyst.invoke(input_schema)

    final_message = response["messages"][-1]

    return {
        "messages": [final_message],
        "details": response["generated_sql_query"]
    }




data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")

def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "sql":
        return "sql_node"
    elif state.route_response == "etl":
        return "etl_node"
    else:
        raise ValueError(f"Invalid route response: {state.route_response}")


data_agent_graph.add_conditional_edges("router_node", route_edge,
                                      {
                                          "sql_node": "sql_node",
                                          "etl_node": "etl_node"
                                      })

data_agent = data_agent_graph.compile()


if __name__ == "__main__":

    # Optional - draws the graph (needs IPython + internet), so it only runs when this file is run directly
    from IPython.display import display, Image
    img = Image(data_agent.get_graph().draw_mermaid_png())
    with open("data_agent_graph.png", "wb") as f:
        f.write(img.data)

    response = data_agent.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
         "route_response": ""}
    )

    print(response)