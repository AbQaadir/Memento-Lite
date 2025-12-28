import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import MementoAgent

def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY in .env file or environment variables.")
        return

    agent = MementoAgent()
    
    print("Welcome to Memento CLI")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nEnter Task > ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                continue
            
            result = agent.run(user_input)
            print("\nFINAL RESULT:")
            print(result)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
