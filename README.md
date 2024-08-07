# MySmplRAG
## Description
MySmplRAG is a user-friendly Retrieval-Augmented Generation (RAG) application featuring a graphical user interface built with Tkinter. This tool leverages ChromaDB for similarity searches and utilizes local LLM and Embedding models via LM Studio, providing a seamless and efficient experience for generating and retrieving relevant information.
## Installation and usage (Windows)

1. **Download and install LM Studio** from [LM Studio](https://lmstudio.ai/)

2. **Set up new environment (optional)**
  ```bash
    conda create --name myrag python=3.12
  ```

3. **Install requirements**
  ```bash
    pip install -r requirements.txt
  ```

4. **LM Studio setup**

   a) Download a LLM and a Embedding model from the 'Search' section
   
   b) In 'Local Server' section load the LLM and the Embeding model
   
   c) Start the local server

6. **Start MySmplRAG**
  ```bash
    python.exe .\AI_RAG_GUI.py
  ```

6. **Usage**

   a) in 'Collections' tab create a collection and add documents to it (at the moment, only .txt files are added and only at the folder level)
   
   b) in 'Main' tab ask the database (i.e. collection) for information present in the loaded files.
   
   c) in 'Configs' tab usage of the application can be configured. More low level configs are in the config.json file

ENJOY!
