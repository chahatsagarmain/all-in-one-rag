import sys
import os
import time
from config.config import Config
from rag_builder.rag_builder import RAGBuilder

# Reconfigure stdout/stderr to use UTF-8 on Windows to avoid UnicodeEncodeError
if sys.platform == 'win32':
    try:
        reconfigure_stdout = getattr(sys.stdout, 'reconfigure', None)
        if reconfigure_stdout is not None:
            reconfigure_stdout(encoding='utf-8')
        reconfigure_stderr = getattr(sys.stderr, 'reconfigure', None)
        if reconfigure_stderr is not None:
            reconfigure_stderr(encoding='utf-8')
    except Exception:
        pass


# ANSI color codes
CLEAR = "\033[H\033[J"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"

# Foreground Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

def print_banner():
    banner_all_in_one = (
        f"{MAGENTA}{BOLD}    ___   __    __        ____        ____  _   __ ______\n"
        f"   /   | / /   / /       /  _/___    / __ \\/ | / // ____/\n"
        f"  / /| |/ /   / /________/ // __ \\  / / / /  |/ // __/   \n"
        f" / ___ / /___/ /___/____/ // / / / / /_/ / /|  // /___   \n"
        f"/_/  |_/_____/_____/   /___/_/ /_/  \\____/_/ |_//_____/{RESET}"
    )
    banner_rag = (
        f"{CYAN}{BOLD}                  ____  ___   ______\n"
        f"                 / __ \\/   | / ____/\n"
        f"                / /_/ / /| |/ / __  \n"
        f"               / _, _/ ___ / /_/ /  \n"
        f"              /_/ |_/_/  |_|\\____/{RESET}"
    )
    
    print(banner_all_in_one)
    print(banner_rag)
    print("\n" + "=" * 60 + "\n")

def get_input(prompt: str, default: str) -> str:
    try:
        user_input = input(f"{BOLD}{prompt}{RESET} {DIM}[{default}]{RESET}: ").strip()
        return user_input if user_input else default
    except KeyboardInterrupt:
        print(f"\n\n{RED}Process interrupted. Goodbye!{RESET}")
        sys.exit(0)

def run_cli():
    # Clear screen
    print(CLEAR, end="")
    
    # Print cool banner
    print_banner()

    print("Current working directory : " + os.getcwd())
    
    # 1. Says lets build your rag first
    print(f"{GREEN}{BOLD}✨ Let's build your RAG pipeline first! ✨{RESET}\n")
    
    # 2. Ask for a path of pdf / markdown / txt
    default_path = "./README.md"
    file_path = get_input("📂 Enter path to PDF / Markdown / TXT file", default_path)
    
    # 3. Select chunking model
    print(f"\n{YELLOW}{BOLD}Select the Chunking Model:{RESET}")
    print(f"  {CYAN}1.{RESET} Fixed Size Chunking")
    print(f"  {CYAN}2.{RESET} Semantic Chunking")
    chunk_choice = get_input("👉 Choice (1 or 2)", "1")
    
    chunk_model = "Fixed Size Chunking" if chunk_choice == "1" else "Semantic Chunking"
    
    # 4. Select embedding way
    print(f"\n{YELLOW}{BOLD}Select the Embedding Method:{RESET}")
    print(f"  {CYAN}1.{RESET} Sentence Transformers (Local)")
    print(f"  {CYAN}2.{RESET} OpenAI Embeddings (API)")
    print(f"  {CYAN}3.{RESET} Static Vectors (Word2Vec)")
    embed_choice = get_input("👉 Choice (1, 2, or 3)", "1")
    
    if embed_choice == "2":
        embed_method = "OpenAI Embeddings"
    elif embed_choice == "3":
        embed_method = "Static Vectors (Word2Vec)"
    else:
        embed_method = "Sentence Transformers"

    # 4b. Select LLM Chat Client
    print(f"\n{YELLOW}{BOLD}Select the Chat Model / LLM Provider:{RESET}")
    print(f"  {CYAN}1.{RESET} OpenAI (gpt-4o-mini)")
    print(f"  {CYAN}2.{RESET} Google Gemini (gemini-1.5-flash)")
    print(f"  {CYAN}3.{RESET} Offline / Mock Mode")
    llm_choice = get_input("👉 Choice (1, 2, or 3)", "1")
    
    if llm_choice == "2":
        llm_model = "Google Gemini"
    elif llm_choice == "3":
        llm_model = "Offline/Mock LLM"
    else:
        llm_model = "OpenAI GPT"

    # Set environment variables for Config loader
    os.environ["CHUNKING_METHOD"] = "fixed" if chunk_choice == "1" else "semantic"
    if embed_choice == "2":
        os.environ["EMBEDDING_METHOD"] = "openai"
        os.environ["EMEDDING_METHOD"] = "openai"
    elif embed_choice == "3":
        os.environ["EMBEDDING_METHOD"] = "static"
        os.environ["EMEDDING_METHOD"] = "static"
    else:
        os.environ["EMBEDDING_METHOD"] = "local"
        os.environ["EMEDDING_METHOD"] = "local"
    
    # Print selection summary
    print(f"\n{GREEN}Configuration Saved:{RESET}")
    print(f"  • Source Path: {WHITE}{file_path}{RESET}")
    print(f"  • Chunking:    {WHITE}{chunk_model}{RESET}")
    print(f"  • Embedding:   {WHITE}{embed_method}{RESET}")
    print(f"  • Chat LLM:    {WHITE}{llm_model}{RESET}\n")

    # 5. Print a message to wait....
    print(f"{YELLOW}🔄 Building your RAG pipeline, please wait...{RESET}", end="", flush=True)
    for _ in range(4):
        time.sleep(0.6)
        print(".", end="", flush=True)
    RAG = RAGBuilder().get_rag_pipeline(file_path=file_path)
    RAG.build()
    time.sleep(2)
    print(f" {GREEN}Done!{RESET}\n")
    
    # Instantiate LLM Chat Client and ChatManager
    from chat.chat_client import OpenAIChatClient, GeminiChatClient, MockChatClient
    from chat.chat_manager import ChatManager
    
    if llm_choice == "2":
        chat_client = GeminiChatClient()
    elif llm_choice == "3":
        chat_client = MockChatClient()
    else:
        chat_client = OpenAIChatClient()
        
    chat_manager = ChatManager(
        rag_pipeline=RAG,
        chat_client=chat_client,
        system_prompt="You are a helpful assistant. Answer the user's queries using the provided context when available."
    )
    
    # 6. Say lets start chatting
    print(f"{GREEN}{BOLD}💬 Let's start chatting!{RESET} {DIM}(Type 'exit' or 'quit' to stop){RESET}\n")
    
    # Chat loop
    while True:
        try:
            query = input(f"{CYAN}{BOLD}You >{RESET} ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print(f"\n{YELLOW}Goodbye!{RESET}")
                break
            
            # Generate actual response
            print(f"{MAGENTA}{BOLD}RAG >{RESET} ", end="", flush=True)
            response = chat_manager.chat(query)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Goodbye!{RESET}")
            break
