## To demonstrate the database language embeddings in implementing documents search with LLM
import newspaper
import json
import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"

url = 'https://www.nytimes.com/2023/05/12/science/saturn-moons-jupiter.html'
  
article = newspaper.Article(url=url, language='en')
article.download()
article.parse()

print(str(article.text))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(article.text)
embeddings = OpenAIEmbeddings()

vectordb = Chroma.from_texts(texts, embeddings)
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever())
query = "Which branch of science this article talks about ?"
answer = qa.run(query)
print(answer)






