from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class MemoryEvent:
    text: str
    timestamp: int  # Abstract time unit (e.g., hours or days from start)
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class BenchmarkTask:
    category: str
    events: List[MemoryEvent]
    query: str
    ground_truth: str

def load_tasks() -> List[BenchmarkTask]:
    tasks = []
    
    # Category 1: Static Facts (Simple Retrieval)
   
    tasks.extend([
        BenchmarkTask(
            category="Static Facts",
            events=[
                MemoryEvent(text="Rachel likes Python", timestamp=1, metadata={"subject": "Rachel", "relation": "likes", "object": "Python"})
            ],
            query="What programming language does Rachel like?",
            ground_truth="Python"
        ),
        BenchmarkTask(
            category="Static Facts",
            events=[
                MemoryEvent(text="The project is called GraphMemory", timestamp=1, metadata={"subject": "project", "relation": "named", "object": "GraphMemory"})
            ],
            query="What is the project called?",
            ground_truth="GraphMemory"
        ),
        BenchmarkTask(
            category="Static Facts",
            events=[
                MemoryEvent(text="John works at OpenAI", timestamp=1, metadata={"subject": "John", "relation": "works_at", "object": "OpenAI"})
            ],
            query="Where does John work?",
            ground_truth="OpenAI"
        )
    ])

   
    # Category 2: Updated/Conflicting Facts
   
    tasks.extend([
        BenchmarkTask(
            category="Conflicting Facts",
            events=[
                MemoryEvent(text="Rachel lives in Arusha", timestamp=1, metadata={"subject": "Rachel", "relation": "lives_in", "object": "Arusha"}),
                MemoryEvent(text="Rachel moved to Dar es Salaam", timestamp=10, metadata={"subject": "Rachel", "relation": "lives_in", "object": "Dar es Salaam"})
            ],
            query="Where does Rachel live currently?",
            ground_truth="Dar es Salaam"
        ),
        BenchmarkTask(
            category="Conflicting Facts",
            events=[
                MemoryEvent(text="The meeting is at 10 AM", timestamp=1, metadata={"subject": "meeting", "relation": "time", "object": "10 AM"}),
                MemoryEvent(text="The meeting was rescheduled to 2 PM", timestamp=5, metadata={"subject": "meeting", "relation": "time", "object": "2 PM"})
            ],
            query="What time is the meeting?",
            ground_truth="2 PM"
        ),
        BenchmarkTask(
            category="Conflicting Facts",
            events=[
                MemoryEvent(text="My favorite color is Blue", timestamp=1, metadata={"subject": "User", "relation": "favorite_color", "object": "Blue"}),
                MemoryEvent(text="I changed my favorite color to Red", timestamp=8, metadata={"subject": "User", "relation": "favorite_color", "object": "Red"})
            ],
            query="What is the user's favorite color?",
            ground_truth="Red"
        ),
        BenchmarkTask(
            category="Conflicting Facts",
            events=[
                MemoryEvent(text="Alice likes Python", timestamp=1, metadata={"subject": "Alice", "relation": "likes", "object": "Python"}),
                MemoryEvent(text="Alice now likes Rust", timestamp=5, metadata={"subject": "Alice", "relation": "likes", "object": "Rust"}),
                MemoryEvent(text="Alice decided Go is actually the best", timestamp=15, metadata={"subject": "Alice", "relation": "likes", "object": "Go"})
            ],
            query="What language does Alice like?",
            ground_truth="Go"
        )
    ])

   
    # Category 3: Long-Term Retention
   
    tasks.extend([
        BenchmarkTask(
            category="Long-Term Retention",
            events=[
                MemoryEvent(text="The master password is 'hunter2'", timestamp=1, metadata={"subject": "password", "relation": "is", "object": "hunter2"}),
                MemoryEvent(text="Bob bought an apple", timestamp=5, metadata={"subject": "Bob", "relation": "bought", "object": "apple"}),
                MemoryEvent(text="Alice went to the store", timestamp=10, metadata={"subject": "Alice", "relation": "went_to", "object": "store"}),
                MemoryEvent(text="Charlie is learning Java", timestamp=15, metadata={"subject": "Charlie", "relation": "learning", "object": "Java"})
            ],
            query="What is the master password?",
            ground_truth="hunter2"
        ),
        BenchmarkTask(
            category="Long-Term Retention",
            events=[
                MemoryEvent(text="System architecture requires microservices", timestamp=1, metadata={"subject": "architecture", "relation": "requires", "object": "microservices"}),
                MemoryEvent(text="We need to update the UI", timestamp=20, metadata={"subject": "UI", "relation": "needs", "object": "update"}),
                MemoryEvent(text="Database is running out of space", timestamp=40, metadata={"subject": "Database", "relation": "status", "object": "full"})
            ],
            query="What does the system architecture require?",
            ground_truth="microservices"
        )
    ])


    # Category 4: Forgetting (Noise/Decay testing)
  
    # To test forgetting, we want to query a concept that was overwritten AND long time ago
    # We will test if the OLD fact is properly forgotten, though this is implicitly measured in Conflicting Facts.
    
  
    # Category 5: Reinforcement (Frequency)
   
    tasks.extend([
        BenchmarkTask(
            category="Reinforcement",
            events=[
                MemoryEvent(text="Eve is studying Biology", timestamp=1, metadata={"subject": "Eve", "relation": "studying", "object": "Biology"}),
                MemoryEvent(text="Eve reviewed her Biology notes", timestamp=10, metadata={"subject": "Eve", "relation": "studying", "object": "Biology"}),
                MemoryEvent(text="Eve has a Biology exam", timestamp=20, metadata={"subject": "Eve", "relation": "studying", "object": "Biology"}),
                MemoryEvent(text="Eve is also taking Math", timestamp=25, metadata={"subject": "Eve", "relation": "studying", "object": "Math"})
            ],
            query="What is Eve's primary subject of study?",
            ground_truth="Biology"
        ),
        BenchmarkTask(
            category="Reinforcement",
            events=[
                MemoryEvent(text="The core value is integrity", timestamp=1, metadata={"subject": "core_value", "relation": "is", "object": "integrity"}),
                MemoryEvent(text="We must always maintain integrity", timestamp=5, metadata={"subject": "core_value", "relation": "is", "object": "integrity"}),
                MemoryEvent(text="Integrity is paramount", timestamp=15, metadata={"subject": "core_value", "relation": "is", "object": "integrity"}),
                MemoryEvent(text="Another value is speed", timestamp=20, metadata={"subject": "value", "relation": "is", "object": "speed"})
            ],
            query="What is the core value?",
            ground_truth="integrity"
        )
    ])

    return tasks
