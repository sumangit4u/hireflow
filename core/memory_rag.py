"""
Conversation memory system for tracking search history and context.
Uses LangChain's memory to maintain session context across searches.
"""

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)

class MemoryRAG:
    """Conversation memory system for maintaining search context and history"""
    
    def __init__(self):
        """Initialize LangChain conversation buffer for search tracking"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def record_search(self, query: str, results_count: int):
        """Store search query and result count in conversation memory"""
        self.memory.chat_memory.add_user_message(f"Search: {query}")
        self.memory.chat_memory.add_ai_message(f"Found {results_count} results")
    
    def record_candidate_view(self, candidate_name: str):
        """Record candidate viewing"""
        self.memory.chat_memory.add_user_message(f"Viewed candidate: {candidate_name}")
        self.memory.chat_memory.add_ai_message("Candidate interaction recorded")
    
    def get_search_history(self) -> list:
        """Get recent search queries"""
        queries = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage) and msg.content.startswith("Search:"):
                queries.append(msg.content.replace("Search: ", ""))
        return queries[-5:]
    
    def get_memory_stats(self) -> dict:
        """Get simple memory stats"""
        return {
            'total_messages': len(self.memory.chat_memory.messages),
            'search_count': len([m for m in self.memory.chat_memory.messages
                               if isinstance(m, HumanMessage) and m.content.startswith("Search:")]),
            'candidate_views': len([m for m in self.memory.chat_memory.messages
                                  if isinstance(m, HumanMessage) and "candidate" in m.content.lower()])
        }

if __name__ == "__main__":
    mem = MemoryRAG()

    print("=== record_search ===")
    mem.record_search("Python developer", 5)
    mem.record_search("senior data engineer with Spark", 3)
    print("Recorded 2 searches.")

    print("\n=== record_candidate_view ===")
    mem.record_candidate_view("Jane Smith")
    print("Recorded candidate view for Jane Smith.")

    print("\n=== get_search_history ===")
    history = mem.get_search_history()
    print("Recent searches:", history)

    print("\n=== get_memory_stats ===")
    stats = mem.get_memory_stats()
    print("Memory stats:", stats)
