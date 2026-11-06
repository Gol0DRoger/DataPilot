from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    response = data_agent.invoke(
        {"messages":[HumanMessage(content='''I want to transform the data stored in the 'C:\\Users\\risha\\Downloads\\Data_agent\\data\\extract\\extracted_data.csv' file 
                     and save the transformed data in the 'C:\\Users\\risha\\Downloads\\Data_agent\\data\\transform' folder in the csv format.
                     The transformation should filter the data to show charizard pokemon only.''')],
         "route_response": ""}
    )

    print(response)