from agent import Agent


def main():
    agent = Agent()
    print("ReAct Agent Started. Type 'exit' to quit.")
    query = input("Enter your query: ")
    response = agent.run(query)
    print("End of execution")


if __name__ == "__main__":
    main()
