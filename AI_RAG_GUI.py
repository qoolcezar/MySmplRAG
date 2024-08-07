import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import webbrowser
import json
from helpers import Chroma_Database
from autogen import AssistantAgent, UserProxyAgent

class MiniRAGTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini RAG tool")
        self.root.minsize(600, 400)
        
        self.create_widgets()
        self.configure_grid()
        self.binding_actions()
        self.init_RAG()
    
    def create_widgets(self):
        # Create a notebook (tab control)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky='nsew')
        
        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text='Main')
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text='Configs')
        self.collections_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.collections_tab, text='Collections')
        
        # Main collection frame
        self.main_collection_frame = ttk.Frame(self.main_tab)
        self.main_collection_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        tk.Label(self.main_collection_frame, text="Search in collection:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.selected_collection = tk.StringVar()
        self.collection_dropdown = ttk.Combobox(self.main_collection_frame, textvariable=self.selected_collection)
        self.collection_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Main response frame
        self.main_response_frame = ttk.Frame(self.main_tab)
        self.main_response_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        tk.Label(self.main_response_frame, text="AI response:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.response_Y_scrollbar = ttk.Scrollbar(self.main_response_frame, orient="vertical")
        self.response_field = tk.Text(self.main_response_frame, height=10, width=50, yscrollcommand=self.response_Y_scrollbar.set)
        self.response_field.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        self.response_Y_scrollbar.config(command=self.response_field.yview)
        self.response_Y_scrollbar.grid(row=1, column=1, sticky="ns")
        
        tk.Label(self.main_response_frame, text="Reference files:").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        self.table_Y_scrollbar = ttk.Scrollbar(self.main_response_frame, orient="vertical")
        self.files_table = ttk.Treeview(self.main_response_frame, columns=("File name", "Relevance"), show='headings', height=2, yscrollcommand=self.table_Y_scrollbar.set)
        self.files_table.heading("File name", text="File name")
        self.files_table.heading("Relevance", text="Relevance")
        self.files_table.grid(row=3, column=0, padx=5, pady=5, sticky='nsew')
        self.files_table.insert("", "end", values=("No data", ""))
        self.table_Y_scrollbar.config(command=self.files_table.yview)
        self.table_Y_scrollbar.grid(row=3, column=1, sticky="ns")
        self.reset_btn = tk.Button(self.main_response_frame, text="Reset", command=self.reset, height=3, width=5)
        self.reset_btn.grid(row=3, column=2)

        # Main ask frame
        self.main_ask_frame = ttk.Frame(self.main_tab)
        self.main_ask_frame.grid(row=2, column=0, padx=5, pady=5, sticky='nsew')
        tk.Label(self.main_ask_frame, text="Search for:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.ask_Y_scrollbar = ttk.Scrollbar(self.main_ask_frame, orient="vertical")
        self.ask_field = tk.Text(self.main_ask_frame, height=3, width=30, yscrollcommand=self.ask_Y_scrollbar.set)
        self.ask_field.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        self.ask_Y_scrollbar.config(command=self.ask_field.yview)
        self.ask_Y_scrollbar.grid(row=1, column=1, sticky="ns")
        self.ask_btn = tk.Button(self.main_ask_frame, text="Ask", command=self.ask, height=3, width=5)
        self.ask_btn.grid(row=1, column=2)

        # Config frame
        self.config_frame = ttk.Frame(self.config_tab)
        self.config_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.optimized_DB_query_var = tk.BooleanVar()
        self.optimized_DB_query_check = tk.Checkbutton(self.config_frame, text="Optimize query", variable=self.optimized_DB_query_var)
        self.optimized_DB_query_check.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.query_nr_results = tk.Scale(self.config_frame, from_=1, to=10, orient='horizontal', label='Query results (document chunks used)')
        self.query_nr_results.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        self.use_llm_response_var = tk.BooleanVar()
        self.use_llm_response_check = tk.Checkbutton(self.config_frame, text="Use LLM interpretation on queried results", variable=self.use_llm_response_var)
        self.use_llm_response_check.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.config_btn = tk.Button(self.config_frame, text="Advance", command=self.open_advance_config)
        self.config_btn.grid(row=3, column=0, sticky='w')

        # Collections frame
        self.edit_collection_frame_1 = ttk.Frame(self.collections_tab)
        self.edit_collection_frame_1.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        tk.Label(self.edit_collection_frame_1, text="Collection:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.edit_selected_collection = tk.StringVar()
        self.edit_collection_dropdown = ttk.Combobox(self.edit_collection_frame_1, textvariable=self.edit_selected_collection)
        self.edit_collection_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        tk.Label(self.edit_collection_frame_1, text="Collection files:").grid(row=1, column=0, padx=5, pady=2, sticky='w')

        self.edit_collection_frame_2 = ttk.Frame(self.collections_tab)
        self.edit_collection_frame_2.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        self.edit_coll_table_Y_scrollbar = ttk.Scrollbar(self.edit_collection_frame_2, orient="vertical")
        self.edit_coll_files_table = ttk.Treeview(self.edit_collection_frame_2, columns=("File name", "Path"), show='headings', height=10, yscrollcommand=self.edit_coll_table_Y_scrollbar.set)
        self.edit_coll_files_table.heading("File name", text="File name")
        self.edit_coll_files_table.heading("Path", text="Path")
        self.edit_coll_files_table.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.edit_coll_table_Y_scrollbar.config(command=self.edit_coll_files_table.yview)
        self.edit_coll_table_Y_scrollbar.grid(row=0, column=1, sticky="ns")

        self.edit_collection_frame_3 = ttk.Frame(self.collections_tab)
        self.edit_collection_frame_3.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        self.add_collection_btn = tk.Button(self.edit_collection_frame_3, text="Add collection", command=self.add_collection )
        self.add_collection_btn.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.delete_collection_btn = tk.Button(self.edit_collection_frame_3, text="Delete collection", command=self.delete_collection )
        self.delete_collection_btn.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.add_doc = tk.Button(self.edit_collection_frame_3, text="Add document(s)", command=self.add_documents)
        self.add_doc.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.delete_doc = tk.Button(self.edit_collection_frame_3, text="Delete document", command=self.delete_documents)
        self.delete_doc.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.check_collection = tk.Button(self.edit_collection_frame_3, text="Recheck docs", command=self.recheck_folder)
        self.check_collection.grid(row=2, column=0, padx=5, pady=5, sticky='w')

    def configure_grid(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_tab.grid_rowconfigure(1, weight=2)
        self.main_tab.grid_rowconfigure(2, weight=1)
        self.main_tab.grid_columnconfigure(0, weight=1)
        self.main_collection_frame.grid_columnconfigure(1, weight=1)
        self.main_response_frame.grid_rowconfigure(1, weight=1)
        self.main_response_frame.grid_rowconfigure(3, weight=1)
        self.main_response_frame.grid_columnconfigure(0, weight=1)
        self.main_ask_frame.grid_rowconfigure(1, weight=1)
        self.main_ask_frame.grid_columnconfigure(0, weight=1)

        self.config_tab.grid_columnconfigure(0, weight=1)
        self.config_frame.grid_columnconfigure(0, weight=1)

        self.collections_tab.grid_rowconfigure(1, weight=1)
        self.collections_tab.grid_columnconfigure(0, weight=1)
        self.edit_collection_frame_1.grid_columnconfigure(1, weight=1)
        self.edit_collection_frame_2.grid_rowconfigure(0, weight=1)
        self.edit_collection_frame_2.grid_columnconfigure(0, weight=1)

    def open_advance_config(self):
        script_dir = os.path.dirname(__file__)
        config_name = 'config.json'
        file_path = os.path.join(script_dir, config_name)
        self.open_file(file_path)  

    def binding_actions(self):
        self.collection_dropdown.bind("<<ComboboxSelected>>", self.load_collection)
        self.files_table.bind("<Double-1>", self.on_file_double_click_1)
        self.edit_coll_files_table.bind("<Double-1>", self.on_file_double_click_2)
        self.edit_collection_dropdown.bind("<<ComboboxSelected>>", self.load_collection_files)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_selected)

    def add_collection(self):
        coll_to_add = self.edit_selected_collection.get()
        if coll_to_add == '':
            messagebox.showwarning("Warning", "Can't have a blank collection name!\nPlease enter a valid name!")
            return
        print(f'Add collection= {coll_to_add}')
        collection_names = [collection.name for collection in self.cdb.client.list_collections()]
        if coll_to_add in collection_names:
            messagebox.showwarning("Warning", f"Collection name '{coll_to_add}' already exists!\nPlease choose another name.")
            return
        collection_names.append(coll_to_add)
        self.collection_dropdown['values'] = collection_names
        self.edit_collection_dropdown['values'] = collection_names
        self.selected_collection.set(coll_to_add)
        self.load_collection(self.selected_collection.get())
        self.load_collection_files(None)

    def delete_collection(self):
        coll_to_delete = self.edit_selected_collection.get()
        collection_names = [collection.name for collection in self.cdb.client.list_collections()]
        if coll_to_delete not in collection_names:
            return
        result = messagebox.askokcancel("Confirmation", f"Are you sure you want to permanently delete collection '{coll_to_delete}'?")
        if not result:
            return
        self.cdb.client.delete_collection(coll_to_delete)
        collection_names.remove(coll_to_delete)
        self.collection_dropdown['values'] = collection_names
        self.edit_collection_dropdown['values'] = collection_names
        if len(collection_names):
            self.selected_collection.set(collection_names[0])
            self.edit_collection_dropdown.set(collection_names[0])
        else:
            self.selected_collection.set('')
            self.edit_collection_dropdown.set('')
        self.load_collection(self.selected_collection.get())
        self.load_collection_files(None)

    def add_documents(self):
        folder_selected = filedialog.askdirectory()
        if not folder_selected:
            return
        collection_to_add_in = self.edit_collection_dropdown.get()
        loaded_collection = self.collection_dropdown.get()
        if collection_to_add_in != loaded_collection:
            self.selected_collection.set(collection_to_add_in)
            self.load_collection(self.selected_collection.get())
        check_result = self.cdb.docs_check_sync(folder_selected)
        print(f"Folder selected: {folder_selected} and found {len(check_result['new_doc_ids'])} txt files to add...")
        self.cdb.add_to_collection()
        self.load_collection_files(None)

    def delete_documents(self):
        items_ids_to_delete = self.edit_coll_files_table.selection()
        if not len(items_ids_to_delete):
            return
        if self.edit_coll_files_table.item(items_ids_to_delete[0], "values")[0] == 'No collection in database!':
            return
        result = messagebox.askokcancel("Confirmation", f"Are you sure you want to remove selected documents from RAG database?")
        if not result:
            return
        ids_to_delete = []
        for each_item in items_ids_to_delete:
            values = self.edit_coll_files_table.item(each_item, "values")
            ids_to_delete += values[2].split()
        self.cdb.delete_from_collection(ids_to_delete)
        self.load_collection_files(None)

    def recheck_folder(self):
        collection_to_add_in = self.edit_collection_dropdown.get()
        loaded_collection = self.collection_dropdown.get()
        if collection_to_add_in != loaded_collection:
            self.selected_collection.set(collection_to_add_in)
            self.load_collection(self.selected_collection.get())
        folders_to_check = set()  # Use a set to store unique values
        for item_id in self.edit_coll_files_table.get_children():
            values = self.edit_coll_files_table.item(item_id, "values")
            folders_to_check.add(values[1])  # Add the value from the second column
        folders_to_check = list(folders_to_check)  # Convert the set back to a list if needed
        check_result = self.cdb.docs_check_sync(folders_to_check)
        if not len(check_result['new_doc_ids']) and not len(check_result['doc_ids_to_delete']):
            messagebox.showwarning("Info", "No change detected!")
        else:
            result = messagebox.askokcancel("Confirmation", f"Change detected, update database?")
            if not result:
                return
            if len(check_result['doc_ids_to_delete']):
                self.cdb.delete_from_collection(check_result['doc_ids_to_delete'])
            if len(check_result['new_doc_ids']):
                self.cdb.add_to_collection()
            self.load_collection_files(None)

    def load_collection(self, collection_name = None):
        self.reset()
        if collection_name == '':
            message = "No collection in the Database !!!\n\nCreate one in the 'Collections' tab!"
            self.set_response_field_text(message)
            return 
        self.cdb.init_collection(collection_name=collection_name)

    def on_tab_selected(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == "Collections":
            if self.edit_selected_collection.get() != self.selected_collection.get():
                self.edit_selected_collection.set(self.selected_collection.get())
            self.load_collection_files(None)

    def load_collection_files(self, _):
        # Clear any existing rows
        self.edit_coll_files_table.delete(*self.edit_coll_files_table.get_children())
        edit_selected_collection = self.edit_selected_collection.get()
        if edit_selected_collection == '':
            self.edit_coll_files_table.insert("","end", values=("No collection in database!",""))
            return
        collection = self.cdb.client.get_collection(name=edit_selected_collection)
        all_doc_chunks = collection.get(include=['metadatas'])
        for idx, each in enumerate(all_doc_chunks['ids']):
            doc_id = each.split('>')[0]
            doc_chunk_ids = [ids for ids in all_doc_chunks['ids'] if ids.split('>')[0] == doc_id]
            all_doc_chunks['metadatas'][idx]['doc_ids'] = doc_chunk_ids
        doc_chunks = [doc for doc in all_doc_chunks['metadatas'] if doc['doc_chunk'] == '0']
        # Populate the table with new data
        for each_doc in doc_chunks:
            directory_path, file_name = os.path.split(each_doc['doc_path'])
            self.edit_coll_files_table.insert("","end", values=(file_name, directory_path, each_doc['doc_ids']))

    def populate_table(self, data_list):
        # Clear any existing rows
        self.files_table.delete(*self.files_table.get_children())
        # Populate the table with new data
        added_doc_paths = []
        for item in data_list:
            doc_path = item['doc_path']
            if doc_path not in added_doc_paths:
                added_doc_paths.append(doc_path)
                file_name = os.path.basename(doc_path)
                self.files_table.insert("", "end", values=(file_name, item['relevance'], item['doc_path']))

    def on_file_double_click_1(self, _):
        if not len(self.files_table.selection()):
            return
        item_id = self.files_table.selection()[0]
        values = self.files_table.item(item_id, "values")
        if values[0] == 'No data':
            return
        file_path = values[2]  # Get the full path
        self.open_file(file_path)

    def on_file_double_click_2(self, _):
        if not len(self.edit_coll_files_table.selection()):
            return
        item_id = self.edit_coll_files_table.selection()[0]
        values = self.edit_coll_files_table.item(item_id, "values")
        file_path = os.path.join(values[1], values[0])
        self.open_file(file_path)    

    def open_file(self, file_path):
        try:
            if os.path.exists(file_path):
                webbrowser.open(file_path)  # Open the file with the default application
            else:
                print(f"File does not exist: {file_path}")
        except Exception as e:
            print(f"An error occurred while opening the file: {e}")

    def ask(self):
        self.set_response_field_text('Processing...\n\nPlease wait!')
        optimized_DB_query = self.optimized_DB_query_var.get()
        use_llm_response = self.use_llm_response_var.get()
        self.cdb.config_json['query_nr_results'] = self.query_nr_results.get()
        user_query = self.ask_field.get("1.0", tk.END).strip()
        if user_query == '':
            self.set_response_field_text('No text in the search field!')
            return
        if optimized_DB_query:
            optimized_DB_query_prompt_template = self.RAG_config['rag_config']['optimized_DB_query_prompt_template']
            prompt_1 = f"{optimized_DB_query_prompt_template}{user_query}"
            chat_history_1 = self.user_proxy.initiate_chat(self.assistant, message=prompt_1, max_turns=1)
            user_query = chat_history_1.chat_history[-1]['content']
        query_results = self.cdb.query_collection(query_texts = user_query)
        # create respond prompt template with each chunk from the resulted vector database query
        chunks = ""
        chunk_count = 0
        chunks_info = []
        for query_count in range(len(query_results['documents'])):
            for query_result_count in range(len(query_results['documents'][query_count])):
                chunk_data = query_results['metadatas'][query_count][query_result_count]
                chunk_data['relevance'] = 1 - query_results['distances'][query_count][query_result_count]
                chunks_info.append(chunk_data)
                chunk_count += 1
                chunks += f"Chunk {chunk_count}:\n{query_results['documents'][query_count][query_result_count]}\n"
        if len(chunks_info) == 0:
            self.files_table.delete(*self.files_table.get_children())
            self.files_table.insert("", "end", values=("No data", ""))
        else:
            self.populate_table(chunks_info)
        if use_llm_response:
            response_prompt_template = self.RAG_config['rag_config']['response_prompt_template']
            prompt_2  = f"{response_prompt_template}{user_query}\n\nCHUNKS:\n\n{chunks}"
            self.assistant.reset()
            chat_history_2 = self.user_proxy.initiate_chat(self.assistant, message=prompt_2, max_turns=1)
            self.set_response_field_text(chat_history_2.summary)
        else:
            self.set_response_field_text('Info found in the below documents.\n\nLLM not selected to interpret it.')

    def reset(self):
        self.response_field.delete('1.0', tk.END)
        self.files_table.delete(*self.files_table.get_children())
        self.files_table.insert("", "end", values=("No data", ""))
        self.ask_field.delete('1.0', tk.END)

    def set_response_field_text(self, text):
        self.response_field.delete('1.0', tk.END)
        self.response_field.insert(tk.END, text)
        self.response_field.update()

    def init_RAG(self):
        try:
            self.RAG_config = self.load_RAG_config()
        except Exception as exp_txt:
            message = f"!!! ERROR loading the config file !!!\n\n{exp_txt}"
            self.set_response_field_text(message)
            return

        chroma_config = self.RAG_config['chroma_config']
        rag_config = self.RAG_config['rag_config']
        self.query_nr_results.set(value=chroma_config['query_nr_results'])
        self.optimized_DB_query_var.set(value=rag_config['optimized_DB_query'])
        self.use_llm_response_var.set(value=rag_config['use_llm_response'])
        os.environ['AUTOGEN_USE_DOCKER'] = rag_config['AUTOGEN_USE_DOCKER']
        llm_agent_config = rag_config['llm_agent_config']
        llm_config = {
            "config_list" :[
                {
                    "model": llm_agent_config['model'],
                    "base_url": llm_agent_config['base_url'],
                    "api_key":llm_agent_config['api_key'],
                },
            ],
            "cache_seed": llm_agent_config['cache_seed'],
        }
        try:
            self.cdb = Chroma_Database(config_json=chroma_config)
            collection_names = [collection.name for collection in self.cdb.client.list_collections()]
            self.edit_collection_dropdown['values'] = collection_names
            self.collection_dropdown['values'] = collection_names
            if chroma_config['default_COLLECTION_NAME'] in collection_names:
                self.selected_collection.set(chroma_config['default_COLLECTION_NAME'])
            elif not len(collection_names):
                pass
            else:
                self.selected_collection.set(collection_names[0])
            self.load_collection(self.selected_collection.get())
        except Exception as exp_txt:
            message = f"!!! ERROR loading Chroma Database !!!\n\n{exp_txt}"
            self.set_response_field_text(message)
            return

        try:
            # Create the agent that uses the LLM.
            self.assistant = AssistantAgent(
                name = "agent", 
                llm_config=llm_config,
                system_message="You are a smart AI")
        except Exception as exp_txt:
            message = f"!!! ERROR loading the AssistantAgent !!!\n\n{exp_txt}"
            self.set_response_field_text(message)
            return
        
        try:
            # Create the agent that represents the user in the conversation.
            self.user_proxy = UserProxyAgent(
                name = "user", 
                code_execution_config=False,
                #llm_config=llm_config,
                #is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
                #human_input_mode="TERMINATE",
                #system_message="You will check the received response from the agent and reply with propper instructions so he can generate the optimal query for a vector database search.",
                )
        except Exception as exp_txt:
            message = f"!!! ERROR loading the UserProxyAgent !!!\n\n{exp_txt}"
            self.set_response_field_text(message)
            return

    def load_RAG_config(self):
        script_dir = os.path.dirname(__file__)
        config_name = 'config.json'
        file_path = os.path.join(script_dir, config_name)
        with open(file_path, 'r') as json_file:
            config_json = json.load(json_file)
        return config_json

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MiniRAGTool(root)
    root.mainloop()
