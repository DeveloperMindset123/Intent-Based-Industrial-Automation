# Create and run agent
from ReActXen.src.reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
from agent_implementation import agent_config

root_agent = create_reactxen_agent(**agent_config)
root_agent.run()


def main():
    print("Hello from intent-implementation-demo!")


if __name__ == "__main__":
    main()
