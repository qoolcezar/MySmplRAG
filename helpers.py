
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions

class WordBasedTextSplitter:
    def __init__(self, max_words_per_chunk, overlap_words, punctuations=None):
        self.max_words_per_chunk = max_words_per_chunk
        self.overlap_words = overlap_words
        self.punctuations = punctuations if punctuations else ['.', ',', '!', '?', ';', ' ']

    def words_count(self, text):
        # Create a regex pattern for splitting by words and the specified punctuation marks
        punctuation_pattern = f"[{''.join(re.escape(p) for p in self.punctuations)}]"
        pattern = rf'\w+|{punctuation_pattern}'

        # Step 1: Split text into words and punctuation marks using the regex pattern
        words_and_punctuations = re.findall(pattern, text)
        # Filter out only words
        words = [token for token in words_and_punctuations if re.match(r'\w+', token)]
        return len(words)

    def split_text(self, text):
        # Get number of words
        num_words = self.words_count(text=text)

        # Step 2: Get the number of characters in the text
        num_chars = len(text)
        
        # Step 3: Determine the average count of characters in a word
        avg_chars_per_word = num_chars / num_words
        
        # Step 4: Determine the inputs for the RecursiveCharacterTextSplitter
        chunk_size = int(self.max_words_per_chunk * avg_chars_per_word)
        overlap_size = int(self.overlap_words * avg_chars_per_word)
        
        # Initialize the RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
        
        # Step 5: Split the text and return the resulting chunks
        chunks = splitter.split_text(text)
        return chunks

def get_file_hash(file_path):
    hash_func = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def get_unique_id_from_path(relative_path):
    path_bytes = relative_path.encode('utf-8')
    hash_func = hashlib.sha256()
    hash_func.update(path_bytes)
    return hash_func.hexdigest()

def read_text_files(init_folder_path):
    """
    Reads all .txt files in the specified folder, accepting both relative and absolute paths, 
    and returns a dictionary with the contents of the files and their absolute paths.

    Args:
    init_folder_path (str): The path to the folder containing .txt files.

    Returns:
    dict: A dictionary with two keys: 'documents' containing a list of the contents of each .txt file,
          and 'paths' containing a list of absolute paths to each .txt file.
    """
    # Convert relative path to absolute path
    folder_path = os.path.abspath(init_folder_path)

    # Check if the directory exists
    if not os.path.exists(folder_path):
        print(f"The directory {folder_path} does not exist.")
        return {'documents': [], 'paths': [], "content_hashes":[], "doc_ids":[]}

    # Check if the path is indeed a directory
    if not os.path.isdir(folder_path):
        print(f"The path {folder_path} is not a directory.")
        return {'documents': [], 'paths': [], "content_hashes":[], "doc_ids":[]}

    # Dictionary to store results
    result = {"documents": [], "paths": [], "content_hashes":[], "doc_ids":[]}

    # Iterate through all files in the specified directory
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            # Construct absolute path to the text file
            file_path = os.path.join(folder_path, filename)
            # Add the file path to the list
            result['paths'].append(file_path)
            result['content_hashes'].append(get_file_hash(file_path))
            relative_file_path = os.path.join(os.path.normpath(init_folder_path), filename)
            result['doc_ids'].append(get_unique_id_from_path(relative_file_path))

            # Read the content of the file
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Add the content to the list
                    result['documents'].append(content)
            except IOError as e:
                print(f"Could not read file {file_path}: {e}")

    return result

def merge_dictionaries(dict1, dict2):
    for key in dict2:
        if key in dict1:
            dict1[key].extend(dict2[key])
        else:
            dict1[key] = dict2[key]
    return dict1

class Chroma_Database:
    def __init__(self, config_json: dict):
        self.config_json = config_json
        self.client = chromadb.PersistentClient(path=config_json['CHROMA_DATA_PATH'])
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name = config_json['OpenAI_embedding_config']['model_name'],
                api_base = config_json['OpenAI_embedding_config']['api_base'],
                api_key = config_json['OpenAI_embedding_config']['api_key']
            )

    def init_collection(self, collection_name = None):
        self.clean_collection_info()
        if collection_name is None:
            collection_name = self.config_json['default_COLLECTION_NAME']
        try:
            # connect to existing db
            self.collection = self.client.get_collection(
                            name=collection_name, 
                            embedding_function=self.openai_ef
                        )
        except:
            # create db
            self.collection = self.client.create_collection(
                            name=collection_name,
                            embedding_function=self.openai_ef,
                            metadata={"hnsw:space": "cosine"},
                        )

    def clean_collection_info(self):
        self.doc_output = None
        self.doc_ids_to_delete = None
        self.new_doc_ids = None

    def docs_check_sync_bk(self, doc_file_path = None):
        if doc_file_path is None:
            self.doc_output = read_text_files(self.config_json['default_doc_files_path'])
        else:
            self.doc_output = read_text_files(doc_file_path)

        content_hashes = self.doc_output['content_hashes']
        doc_ids = self.doc_output['doc_ids']
        # determine what to add, update or delete in DB
        stored_docs = self.collection.get()
        stored_ids = stored_docs['ids']
        # what to delete
        stored_doc_ids = []
        doc_ids_2del = []
        for each_stored_idx, stored_id in enumerate(stored_ids):
            stored_doc_id = stored_id.split('>', 1)[0]
            stored_doc_ids.append(stored_doc_id)
            if stored_doc_id not in doc_ids:
                doc_ids_2del.append(stored_id)
            else:
                # what has changed
                stored_doc_hash = stored_docs['metadatas'][each_stored_idx]['doc_hash']
                if stored_doc_hash not in content_hashes:
                    doc_ids_2del.append(stored_id)
        # what to add
        after_del_stored_doc_ids =  [item for item in stored_ids if item not in doc_ids_2del]
        after_del_stored_doc_ids = [s.split('>', 1)[0] for s in after_del_stored_doc_ids]
        new_doc_ids = [item for item in doc_ids if item not in after_del_stored_doc_ids]
        self.doc_ids_to_delete = doc_ids_2del
        self.new_doc_ids = new_doc_ids
        return {'new_doc_ids': new_doc_ids, 'doc_ids_to_delete': doc_ids_2del}

    def docs_check_sync(self, doc_file_path = None):
        if doc_file_path is None:
            self.doc_output = read_text_files(self.config_json['default_doc_files_path'])
        else:
            if isinstance(doc_file_path, list):
                self.doc_output = {}
                for each_path in doc_file_path:
                    doc_output = read_text_files(each_path)
                    self.doc_output = merge_dictionaries(self.doc_output, doc_output)
            else:
                self.doc_output = read_text_files(doc_file_path)

        content_hashes = self.doc_output['content_hashes']
        doc_ids = self.doc_output['doc_ids']
        # determine what to add, update or delete in DB
        stored_docs = self.collection.get()
        stored_ids = stored_docs['ids']
        # what to delete
        stored_doc_ids = []
        doc_ids_2del = []
        for each_stored_idx, stored_id in enumerate(stored_ids):
            stored_doc_id = stored_id.split('>', 1)[0]
            stored_doc_ids.append(stored_doc_id)
            if stored_doc_id not in doc_ids:
                doc_ids_2del.append(stored_id)
            else:
                # what has changed
                stored_doc_hash = stored_docs['metadatas'][each_stored_idx]['doc_hash']
                if stored_doc_hash not in content_hashes:
                    doc_ids_2del.append(stored_id)
        # what to add
        after_del_stored_doc_ids =  [item for item in stored_ids if item not in doc_ids_2del]
        after_del_stored_doc_ids = [s.split('>', 1)[0] for s in after_del_stored_doc_ids]
        new_doc_ids = [item for item in doc_ids if item not in after_del_stored_doc_ids]
        self.doc_ids_to_delete = doc_ids_2del
        self.new_doc_ids = new_doc_ids
        return {'new_doc_ids': new_doc_ids, 'doc_ids_to_delete': doc_ids_2del}

    def delete_from_collection(self, ids_list = None):
        if ids_list is None:
            ids_list = self.doc_ids_to_delete
        if len(ids_list):
            # Delete from collection DB
            self.collection.delete(ids_list)
    
    def add_to_collection(self, ids_list = None):
        if ids_list is None:
            ids_list = self.new_doc_ids
        doc_ids = self.doc_output['doc_ids']
        content_hashes = self.doc_output['content_hashes']
        documents = self.doc_output['documents']
        doc_paths = self.doc_output['paths']
        # Initialize the WordBasedTextSplitter and add to DB
        splitter = WordBasedTextSplitter(max_words_per_chunk=self.config_json['add_to_collection_config']['max_words_per_chunk'],
                                         overlap_words=self.config_json['add_to_collection_config']['overlap_words'])
        for each_doc_id in ids_list:
            each_doc_idx = doc_ids.index(each_doc_id)
            print(f"Adding/updating document with lenght= {splitter.words_count(documents[each_doc_idx])}")
            # Split the sample text into smaller chunks
            chunks = splitter.split_text(documents[each_doc_idx])
            ids=[f"{doc_ids[each_doc_idx]}>{i}" for i in range(len(chunks))]
            metadatas=[{"doc_path": f"{doc_paths[each_doc_idx]}",
                        "doc_chunk": f"{i}",
                        "doc_hash": f"{content_hashes[each_doc_idx]}"} for i in range(len(chunks))]
            # Add to collection DB
            self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    def query_collection(self, query_texts:str = '', texts_delimiter = '|'):
        query_texts = query_texts.split(texts_delimiter)
        query_results = self.collection.query(
                            query_texts=query_texts,
                            include=["documents", "metadatas", "distances"],
                            n_results=self.config_json['query_nr_results'] # for each query
                        )
        return query_results
