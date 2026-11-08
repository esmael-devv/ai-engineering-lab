
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define the prompt
prompt = ChatPromptTemplate.from_template(
    "Explain the following AI concept to a beginner in 3 sentences: {topic}"
)

# Step 2: Configure the AI model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)

# Step 3: Configure the output parser
parser = StrOutputParser()

# Step 4: Connect everything into a chain
chain = prompt | model | parser

# Step 5: Execute the chain
response = chain.invoke({"topic": "LangChain"})

print(response)
