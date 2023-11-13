rag_prompt = """Create an informative and comprehensive answer for a given question based solely on the given 
documents. You must only use information from the given documents. Use an unbiased and journalistic tone. Do not 
repeat text. Cite the documents using [Document name] notation. If multiple documents contain the answer, cite those 
documents like ‘as stated in [Document name 1], [Document name 2], etc.’. You must include citations in your answer. 
If the documents do not contain the answer to the question, say that  ‘answering is not possible given the available 
information.’ {context}"""
