import streamlit as st
import os
from dotenv import load_dotenv

from langchain_litellm import ChatLiteLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Web search imports
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    st.warning("DuckDuckGo search not available. Install with: pip install duckduckgo-search")

load_dotenv()


def load_embeddings_and_db():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.path.exists("vectorstore"):
            db = Chroma(persist_directory="vectorstore", embedding_function=embedding_model)
        else:
            st.warning("Vector store not found at 'vectorstore'. Please ensure your documents are indexed.")
            db = None
        return embedding_model, db
    except Exception as e:
        st.error(f"Error loading embeddings or database: {str(e)}")
        return None, None

embedding_model, db = load_embeddings_and_db()
retriever = db.as_retriever() if db else None

# Inititialize LLM
@st.cache_resource
def load_llm():
    try:
        return ChatLiteLLM(
            model="openrouter/mistralai/mistral-small-3.2-24b-instruct:free",
            temperature=0.25
        )
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

llm = load_llm()

def clean_web_query(q):
    ignore = {"quel","quels","quelle","quelles"}
    return " ".join([w for w in q.split() if w.lower() not in ignore])

# Enhanced web search
def enhanced_web_search(query: str):
    if not DDGS_AVAILABLE:
        return {'results': "Recherche web non disponible.", 'sources': [], 'success': False, 'query_used': query}
    
    cleaned_query = clean_web_query(query)
    # Add Quebec context to search queries for better localized results
    quebec_enhanced_query = f"{cleaned_query} Québec Canada"
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(quebec_enhanced_query, max_results=5))
        if results:
            sources = []
            results_text = ""
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Résultat Web')
                url = result.get('href', '')
                snippet = result.get('body', '')
                sources.append({'title': title[:100], 'url': url, 'snippet': snippet[:300]})
                results_text += f"{i}. {title}\n{snippet}\nSource: {url}\n\n"
            return {'results': results_text, 'sources': sources, 'success': True, 'query_used': quebec_enhanced_query}
        else:
            return {'results': "Aucun résultat trouvé.", 'sources': [], 'success': False, 'query_used': quebec_enhanced_query}
    except Exception as e:
        return {'results': f"Erreur: {str(e)}", 'sources': [], 'success': False, 'query_used': quebec_enhanced_query}

# Function to analyze which sources were actually used 
def analyze_used_sources(answer: str, context_docs: list) -> list:
    """Analyze which document sources were actually used in the answer"""
    if not context_docs or not answer or not llm:
        return []
    
    used_sources = []
    
    # Create a prompt to identify which sources were actually used
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Tu es un analyseur qui détermine quels documents ont été utilisés pour générer une réponse. "
         "Analyse la réponse et les documents fournis. Pour chaque document, détermine s'il a été "
         "réellement utilisé pour créer la réponse (pas seulement mentionné). "
         "Réponds avec une liste de numéros (1, 2, 3, etc.) des documents qui ont été utilisés, "
         "ou 'AUCUN' si aucun document n'a été utilisé. "
         "Par exemple: '1,3' si les documents 1 et 3 ont été utilisés."),
        ("human", 
         "Réponse générée: {answer}\n\n"
         "Documents disponibles:\n{docs_text}\n\n"
         "Quels documents (numéros) ont été utilisés pour cette réponse?")
    ])
    
    # Prepare documents text for analysis
    docs_text = ""
    for i, doc in enumerate(context_docs, 1):
        docs_text += f"Document {i}: {doc.page_content[:200]}...\n\n"
    
    try:
        analysis_response = llm.invoke(analysis_prompt.format_messages(
            answer=answer,
            docs_text=docs_text
        ))
        
        used_indices = analysis_response.content.strip()
        print(f"AI determined used sources: {used_indices}")
        
        if used_indices and used_indices != "AUCUN" and used_indices.upper() != "AUCUN":
            # Parse the response to get document indices
            try:
                indices = [int(x.strip()) for x in used_indices.split(',') if x.strip().isdigit()]
                for idx in indices:
                    if 1 <= idx <= len(context_docs):
                        doc = context_docs[idx-1]
                        filename = os.path.basename(doc.metadata.get("source", "inconnu"))
                        used_sources.append({
                            "filename": filename,
                            "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                            "full_content": doc.page_content
                        })
            except (ValueError, IndexError) as e:
                print(f"Error parsing used sources: {e}")
                # Fallback: show first 2 sources if we have a substantial answer
                if len(answer) > 200:
                    unique_sources = {}
                    for i, doc in enumerate(context_docs[:2]):
                        filename = os.path.basename(doc.metadata.get("source", "inconnu"))
                        if filename not in unique_sources:
                            unique_sources[filename] = doc
                            used_sources.append({
                                "filename": filename,
                                "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                                "full_content": doc.page_content
                            })
        else:
            # If AI says AUCUN but we have a good answer from documents, show some sources
            if len(answer) > 200 and context_docs:
                print("AI said AUCUN but answer is substantial - showing top sources")
                unique_sources = {}
                for i, doc in enumerate(context_docs[:2]):
                    filename = os.path.basename(doc.metadata.get("source", "inconnu"))
                    if filename not in unique_sources:
                        unique_sources[filename] = doc
                        used_sources.append({
                            "filename": filename,
                            "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                            "full_content": doc.page_content
                        })
        
    except Exception as e:
        print(f"Error in source analysis: {e}")
        # Fallback: return top 2 sources if we have a good answer
        if len(answer) > 200 and context_docs:
            unique_sources = {}
            for doc in context_docs[:2]:
                filename = os.path.basename(doc.metadata.get("source", "inconnu"))
                if filename not in unique_sources:
                    unique_sources[filename] = doc
                    used_sources.append({
                        "filename": filename,
                        "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                        "full_content": doc.page_content
                    })
    
    return used_sources

# Function to check if documents are actually relevant to the question
def check_document_relevance(question: str, context_docs: list) -> bool:
    """Check if documents are actually relevant to answer the question"""
    if not context_docs or len(context_docs) == 0 or not llm:
        return False
    
    # Use AI to check if documents contain relevant information
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Tu es un évaluateur qui détermine si des documents contiennent des informations pertinentes pour répondre à une question. "
         "Analyse la question et le contenu des documents. "
         "Réponds UNIQUEMENT par 'RELEVANT' si les documents contiennent des informations spécifiques et utiles pour répondre à la question, "
         "ou 'NON_RELEVANT' si les documents ne contiennent que des mentions superficielles ou ne sont pas liés au sujet principal."),
        ("human", 
         "Question: {question}\n"
         "Contenu des documents:\n{docs_content}\n\n"
         "Les documents contiennent-ils des informations pertinentes pour répondre à cette question?")
    ])
    
    # Prepare documents content for analysis
    docs_content = ""
    for i, doc in enumerate(context_docs[:3], 1):  # Check first 3 docs
        docs_content += f"Document {i}: {doc.page_content[:200]}...\n\n"
    
    try:
        relevance_response = llm.invoke(relevance_prompt.format_messages(
            question=question,
            docs_content=docs_content
        ))
        
        evaluation = relevance_response.content.strip().upper()
        is_relevant = "RELEVANT" in evaluation and "NON_RELEVANT" not in evaluation
        
        print(f"Document relevance check: {evaluation} -> {is_relevant}")
        return is_relevant
        
    except Exception as e:
        print(f"Error in document relevance check: {e}")
        # Fallback: if documents are too short, consider not relevant
        total_content_length = sum(len(doc.page_content) for doc in context_docs)
        return total_content_length > 500  # Need substantial content

def evaluate_document_answer(question: str, answer: str, context_docs: list) -> dict:
    """Evaluate if the document-based answer is sufficient or if web search is needed"""
    
    if not context_docs or len(context_docs) == 0:
        return {"sufficient": False, "reason": "no documents found"}
    
    # Check total content length
    total_content_length = sum(len(doc.page_content) for doc in context_docs)
    if total_content_length < 50:
        return {"sufficient": False, "reason": "insufficient document content"}
    
    if not llm:
        # Fallback without LLM
        is_sufficient = len(answer) > 100 and total_content_length > 200
        return {"sufficient": is_sufficient, "reason": f"fallback evaluation"}
    
    # Use AI to evaluate if the answer is sufficient
    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Tu es un évaluateur qui détermine si une réponse basée sur des documents est suffisante. "
         "Analyse la question, la réponse générée, et les documents utilisés. "
         "Détermine si la réponse répond adequatement à la question ou si elle est trop générale/vague. "
         "Réponds UNIQUEMENT par 'SUFFICIENT' si la réponse est satisfaisante, "
         "ou 'INSUFFICIENT' si la réponse est trop vague, générale, ou ne répond pas à la question spécifique."),
        ("human", 
         "Question: {question}\n"
         "Réponse générée: {answer}\n"
         "Documents disponibles: {docs_summary}\n\n"
         "Cette réponse est-elle suffisante pour répondre à la question?")
    ])
    
    # Create summary of documents
    docs_summary = f"{len(context_docs)} documents trouvés avec {total_content_length} caractères au total"
    
    try:
        evaluation_response = llm.invoke(evaluation_prompt.format_messages(
            question=question,
            answer=answer,
            docs_summary=docs_summary
        ))
        
        evaluation = evaluation_response.content.strip().upper()
        is_sufficient = "SUFFICIENT" in evaluation
        
        return {
            "sufficient": is_sufficient, 
            "reason": f"AI evaluation: {evaluation}",
            "answer_length": len(answer),
            "doc_count": len(context_docs),
            "total_content": total_content_length
        }
        
    except Exception as e:
        print(f"Error in answer evaluation: {e}")
        is_sufficient = len(answer) > 100 and total_content_length > 200
        return {
            "sufficient": is_sufficient, 
            "reason": f"fallback heuristic: answer_len={len(answer)}, content_len={total_content_length}"
        }

# Session-based message history
def get_session_history(session_id: str):
    if "chat_message_history" not in st.session_state:
        st.session_state.chat_message_history = ChatMessageHistory()
    return st.session_state.chat_message_history

# Simplified intelligent decision-making
def get_intelligent_decision(question: str, chat_history) -> dict:
    """Simplified decision making - let AI evaluate in memory section"""
    
    question_lower = question.lower()
    
    # Get recent conversation context
    recent_context = ""
    if chat_history and len(chat_history.messages) > 0:
        recent_messages = chat_history.messages[-6:]
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                recent_context += f"{msg.content[:150]} "
    
    # 1. DIRECT CONVERSATION REFERENCES - Always memory first
    memory_indicators = [
        "question d'avant", "question précédente", "tu as dit", "vous avez dit",
        "notre conversation", "plus tôt", "tantôt", "dit avant", "parlé de",
        "conversation précédente", "tu m'as dit", "vous m'avez dit"
    ]
    
    if any(indicator in question_lower for indicator in memory_indicators):
        return {"type": "memory", "reason": "direct conversation reference", "enhanced_query": question}
    
    # 2. CONTEXTUAL/CONVERSATION INDICATORS - Let AI decide intelligently
    contextual_indicators = ["leur", "leurs", "sa", "son", "ses", "ce", "cette", "cela", "ça", "ils", "elles"]
    
    # Use AI to determine if this is a conversation/memory question
    if any(indicator in question_lower for indicator in contextual_indicators) and recent_context:
        return {"type": "memory", "reason": "contextual reference - try memory first", "enhanced_query": question}
    
    # Check if question seems to reference conversation using AI
    if recent_context and len(question.split()) < 10 and llm:  # Short questions are more likely to be conversational
        # Use a simple AI check for conversation references
        conversation_check_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Tu es un analyseur qui détermine si une question fait référence à une conversation précédente. "
             "Réponds UNIQUEMENT par 'CONVERSATION' si la question demande quelque chose sur la conversation précédente "
             "(comme 'ma dernière question', 'qu'est-ce que j'ai dit', 'de quoi on parlait', etc.) "
             "ou 'NOUVELLE' si c'est une nouvelle question indépendante."),
            ("human", "Question: {question}\nContexte récent: {context}\n\nCette question fait-elle référence à la conversation?")
        ])
        
        try:
            conv_response = llm.invoke(conversation_check_prompt.format_messages(
                question=question,
                context=recent_context[:200]
            ))
            
            if "CONVERSATION" in conv_response.content.upper():
                return {"type": "memory", "reason": "AI detected conversation reference", "enhanced_query": question}
        except:
            pass  # If AI check fails, continue with normal flow
    
    # 3. CURRENT/RECENT INFO - Direct to web
    current_info_indicators = [
        "2024", "2025", "récent", "actuel", "nouveau", "nouvelles", "aujourd'hui",
        "cette année", "cette semaine", "ce mois", "dernièrement", "maintenant",
        "actualité", "news"
    ]
    
    if any(indicator in question_lower for indicator in current_info_indicators):
        return {"type": "web", "reason": "current information needed", "enhanced_query": question}
    
    # 4. Default to documents for general knowledge
    return {"type": "documents", "reason": "general knowledge question", "enhanced_query": question}

# Enhanced response system with improved decision making
def get_enhanced_response(question: str, chat_history, session_id: str) -> dict:
    """Enhanced response system with improved decision making and Quebec context"""
    
    if not llm:
        return {
            "answer": "Désolé, le système LLM n'est pas disponible.",
            "context": [],
            "used_web": False,
            "web_sources": [],
            "used_memory": False,
            "doc_sources": [],
            "error": True
        }
    
    # Get chat history for context
    chat_hist = get_session_history(session_id)
    
    # Step 1: Use AI to intelligently decide how to handle this question
    decision = get_intelligent_decision(question, chat_hist)
    decision_type = decision.get("type", "documents")
    enhanced_query = decision.get("enhanced_query", question)
    reason = decision.get("reason", "no reason provided")
    
    print(f"AI Decision: {decision_type} - {reason}")
    print(f"Enhanced query: {enhanced_query}")
    
    # Step 2: Handle based on AI decision
    if decision_type == "memory":
        print("Using memory/conversation mode - AI will evaluate if external help needed")
        
        # First: Let AI evaluate if it can answer from memory or needs external help
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Tu es un assistant québécois qui évalue si une question peut être répondue avec l'historique de conversation disponible. "
             "Tu réponds TOUJOURS en français et avec le contexte du Québec. "
             "Analyse l'historique de la conversation et la question posée. "
             "Réponds UNIQUEMENT par 'MEMORY_SUFFICIENT' si tu peux répondre complètement avec l'historique, "
             "ou 'NEED_EXTERNAL_HELP' si tu as besoin de rechercher des informations supplémentaires pour répondre. "
             "Par exemple, si on parle d'astronomes dans l'historique et qu'on demande 'quels sont leurs salaires', "
             "tu devrais répondre 'NEED_EXTERNAL_HELP' car les salaires ne sont pas mentionnés dans l'historique."),
            MessagesPlaceholder("chat_history"),
            ("human", "Question à évaluer: {input}\n\nPeux-tu répondre à cette question avec l'historique de conversation disponible?")
        ])
        
        evaluation_response = llm.invoke(evaluation_prompt.format_messages(
            chat_history=chat_hist.messages,
            input=question
        ))
        
        evaluation = evaluation_response.content.strip().upper()
        print(f"AI Evaluation: {evaluation}")
        
        if "NEED_EXTERNAL_HELP" in evaluation:
            print("AI determined external help needed - switching to web search")
            
            # Extract context and create enhanced query for web search
            context_extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "Tu es un assistant québécois qui répond TOUJOURS en français. "
                 "Extrait le contexte de la conversation et reformule la question pour une recherche web efficace au Québec. "
                 "Par exemple, si on a parlé d'astronomes et qu'on demande 'leurs salaires', "
                 "reformule en 'salaire des astronomes Québec'. "
                 "Réponds UNIQUEMENT avec la requête de recherche reformulée, en français."),
                MessagesPlaceholder("chat_history"),
                ("human", "Question originale: {input}\n\nReformule cette question pour une recherche web au Québec:")
            ])
            
            context_response = llm.invoke(context_extraction_prompt.format_messages(
                chat_history=chat_hist.messages,
                input=question
            ))
            
            enhanced_query = context_response.content.strip()
            print(f"Enhanced query for web search: {enhanced_query}")
            
            # Perform web search with enhanced query
            web_data = enhanced_web_search(enhanced_query)
            web_results = web_data['results']
            web_sources = web_data['sources']
            search_success = web_data.get('success', False)
            
            if search_success and web_sources:
                web_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "Tu es un assistant expert québécois qui répond TOUJOURS en français. "
                     "Tu te trouves au Québec et tu donnes des informations dans le contexte québécois. "
                     "Utilise l'historique de conversation pour le contexte ET les résultats de recherche web "
                     "pour répondre à la question de façon complète. "
                     "Privilégie les informations spécifiques au Québec quand c'est pertinent. "
                     "Assure-toi de répondre uniquement en français."),
                    MessagesPlaceholder("chat_history"),
                    ("human", 
                     f"Question originale: {question}\n"
                     f"Résultats de recherche web:\n{web_results}\n\n"
                     f"Réponds à la question en français en utilisant le contexte de notre conversation "
                     f"et les informations web pertinentes, avec une perspective québécoise.")
                ])
                
                web_response = llm.invoke(web_prompt.format_messages(chat_history=chat_hist.messages))
                
                return {
                    "answer": web_response.content,
                    "context": [],
                    "used_web": True,
                    "web_sources": web_sources,
                    "query_used": web_data['query_used'],
                    "doc_sources": [],
                    "used_memory": False,
                    "ai_switched_to_web": True
                }
            else:
                print("Web search failed, falling back to memory-only response")
        
        # If MEMORY_SUFFICIENT or web search failed, use memory-only response
        print("Using memory-only response")
        conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Tu es un assistant québécois qui répond TOUJOURS UNIQUEMENT en français. "
             "Tu te trouves au Québec et tu donnes des réponses dans le contexte québécois. "
             "L'utilisateur fait référence à votre conversation précédente. "
             "Utilise UNIQUEMENT l'historique de la conversation pour donner une réponse précise et détaillée. "
             "Si tu ne trouves pas l'information dans l'historique, dis-le clairement. "
             "IMPORTANT: Réponds TOUJOURS en français, jamais en anglais."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        response = llm.invoke(conversation_prompt.format_messages(
            chat_history=chat_hist.messages,
            input=question
        ))
        
        return {
            "answer": response.content,
            "context": [],
            "used_web": False,
            "web_sources": [],
            "used_memory": True,
            "doc_sources": [],
            "conversation_based": True
        }
    
    elif decision_type == "web":
        print("Using web search mode")
        web_data = enhanced_web_search(enhanced_query)
        web_results = web_data['results']
        web_sources = web_data['sources']
        search_success = web_data.get('success', False)
        
        if search_success and web_sources:
            # Create a better web prompt 
            web_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "Tu es un assistant expert québécois qui répond en français. "
                 "Tu te trouves au Québec et tu donnes des informations dans le contexte québécois. "
                 "Utilise les résultats de recherche web fournis pour répondre à la question. "
                 "Si les résultats ne sont pas pertinents pour la question, dis-le clairement. "
                 "Privilégie les informations spécifiques au Québec quand c'est disponible. "
                 "Donne une réponse claire et informative basée uniquement sur les informations web pertinentes."),
                MessagesPlaceholder("chat_history"),
                ("human", 
                 f"Question originale: {question}\n"
                 f"Résultats de recherche web:\n{web_results}\n\n"
                 f"Réponds à la question en utilisant les informations web pertinentes, "
                 f"avec une perspective québécoise. Si les résultats ne répondent pas à la question, dis-le clairement.")
            ])
            
            web_response = llm.invoke(web_prompt.format_messages(chat_history=chat_hist.messages))
            
            return {
                "answer": web_response.content,
                "context": [],
                "used_web": True,
                "web_sources": web_sources,
                "query_used": web_data['query_used'],
                "doc_sources": []
            }
        else:
            print("Web search failed, falling back to documents...")
            # Fall through to document search
    
    # Step 3: Try document search 
    if retriever and conversational_rag_chain:
        try:
            # Use enhanced query for better retrieval
            result = conversational_rag_chain.invoke(
                {"input": enhanced_query},
                config={"configurable": {"session_id": session_id}}
            )
            
            answer = result["answer"]
            context_docs = result.get("context", [])
            
            print(f"Document search: {len(context_docs)} docs found")
            
            # First check if documents are actually relevant to the question
            if context_docs and len(context_docs) > 0:
                documents_relevant = check_document_relevance(question, context_docs)
                
                if not documents_relevant:
                    print("Documents not relevant to question - switching to web search")
                    
                    # Documents not relevant, try web search
                    web_data = enhanced_web_search(enhanced_query)
                    web_results = web_data['results']
                    web_sources = web_data['sources']
                    search_success = web_data.get('success', False)
                    
                    if search_success and web_sources:
                        web_prompt = ChatPromptTemplate.from_messages([
                            ("system", 
                             "Tu es un assistant expert québécois qui répond en français. "
                             "Tu te trouves au Québec et tu donnes des informations dans le contexte québécois. "
                             "Les documents locaux ne contenaient pas d'information pertinente, donc tu utilises "
                             "les résultats de recherche web pour répondre à la question. "
                             "Privilégie les informations spécifiques au Québec quand c'est disponible. "
                             "Donne une réponse claire et informative."),
                            MessagesPlaceholder("chat_history"),
                            ("human", 
                             f"Question: {question}\n"
                             f"Résultats de recherche web:\n{web_results}\n"
                             f"Réponds directement en français avec une perspective québécoise en utilisant les informations pertinentes.")
                        ])
                        
                        web_response = llm.invoke(web_prompt.format_messages(chat_history=chat_hist.messages))
                        return {
                            "answer": web_response.content,
                            "context": [],
                            "used_web": True,
                            "web_sources": web_sources,
                            "query_used": web_data['query_used'],
                            "doc_sources": [],
                            "docs_not_relevant": True
                        }
                    else:
                        print("Web search failed after irrelevant documents")
                        # Even web search failed, return minimal response
                        return {
                            "answer": "Je ne trouve pas d'informations suffisamment pertinentes pour répondre à cette question de manière précise. Pourrais-tu reformuler ta question ou être plus spécifique?",
                            "context": [],
                            "used_web": False,
                            "web_sources": [],
                            "used_memory": False,
                            "doc_sources": [],
                            "no_sufficient_info": True
                        }
                
                # Documents are relevant, now evaluate if answer is sufficient
                evaluation = evaluate_document_answer(question, answer, context_docs)
                print(f"Document answer evaluation: {evaluation}")
                
                if evaluation["sufficient"]:
                    print("Using document-based answer")
                    
                    # Analyze which sources were actually used
                    used_doc_sources = analyze_used_sources(answer, context_docs)
                    
                    print(f"Used document sources: {len(used_doc_sources)} out of {len(context_docs)}")
                    
                    return {
                        "answer": answer,
                        "context": context_docs,
                        "used_web": False,
                        "web_sources": [],
                        "used_memory": False,
                        "doc_sources": used_doc_sources
                    }
                else:
                    print(f"Document answer insufficient: {evaluation['reason']}")
                    print("Switching to web search...")
                    
                    # Documents didn't provide good answer, try web search
                    web_data = enhanced_web_search(enhanced_query)
                    web_results = web_data['results']
                    web_sources = web_data['sources']
                    search_success = web_data.get('success', False)
                    
                    if search_success and web_sources:
                        web_prompt = ChatPromptTemplate.from_messages([
                            ("system", 
                             "Tu es un assistant expert québécois qui répond en français. "
                             "Tu te trouves au Québec et tu donnes des informations dans le contexte québécois. "
                             "Les documents locaux ne contenaient pas d'information suffisante, donc tu utilises "
                             "les résultats de recherche web pour répondre à la question. "
                             "Privilégie les informations spécifiques au Québec quand c'est disponible. "
                             "Donne une réponse claire et informative."),
                            MessagesPlaceholder("chat_history"),
                            ("human", 
                             f"Question: {question}\n"
                             f"Résultats de recherche web:\n{web_results}\n"
                             f"Réponds directement en français avec une perspective québécoise en utilisant les informations pertinentes.")
                        ])
                        
                        web_response = llm.invoke(web_prompt.format_messages(chat_history=chat_hist.messages))
                        return {
                            "answer": web_response.content,
                            "context": [],
                            "used_web": True,
                            "web_sources": web_sources,
                            "query_used": web_data['query_used'],
                            "doc_sources": [],
                            "docs_insufficient": True
                        }
            
            # No documents found, try web search
            print("No documents found, trying web search...")
            web_data = enhanced_web_search(enhanced_query)
            web_results = web_data['results']
            web_sources = web_data['sources']
            search_success = web_data.get('success', False)
            
            if search_success and web_sources:
                web_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "Tu es un assistant expert québécois qui répond en français. "
                     "Tu te trouves au Québec et tu donnes des informations dans le contexte québécois. "
                     "Aucun document local n'a été trouvé, donc tu utilises les résultats de recherche web. "
                     "Privilégie les informations spécifiques au Québec quand c'est disponible."),
                    MessagesPlaceholder("chat_history"),
                    ("human", 
                     f"Question: {question}\n"
                     f"Résultats de recherche web:\n{web_results}\n"
                     f"Réponds directement en français avec une perspective québécoise.")
                ])
                
                web_response = llm.invoke(web_prompt.format_messages(chat_history=chat_hist.messages))
                return {
                    "answer": web_response.content,
                    "context": [],
                    "used_web": True,
                    "web_sources": web_sources,
                    "query_used": web_data['query_used'],
                    "doc_sources": [],
                    "no_docs_found": True
                }
                
        except Exception as e:
            print(f"Error in document search: {e}")
    
    # Step 4: Final fallback 
    print("Using final fallback - this should be rare")
    return {
        "answer": "Je rencontre des difficultés techniques pour répondre à cette question. Pourrais-tu essayer de reformuler ta question?",
        "context": [],
        "used_web": False,
        "web_sources": [],
        "used_fallback": True,
        "doc_sources": []
    }

# Create contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is. "
    "Pay special attention to contextual references like 'leur', 'sa', 'ce', etc."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Initialize components only if prerequisites are available
history_aware_retriever = None
conversational_rag_chain = None

if llm and retriever:
    # Create history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Enhanced answer prompt 
    qa_system_prompt = (
        "Tu es un assistant expert québécois qui répond toujours uniquement en français. "
        "Tu te trouves au Québec et tu donnes des informations dans le contexte québécois "
        "(système d'éducation du Québec, salaires québécois, lois québécoises, culture québécoise, etc.). "
        "Si la question concerne l'historique de la conversation ou fait référence à un contexte précédent, "
        "utilise le contexte de la conversation pour donner une réponse pertinente. "
        "Pour les questions générales, utilise tes connaissances et les documents pertinents avec une perspective québécoise. "
        "Pour les questions sur des informations actuelles ou récentes, privilégie "
        "les sources externes si elles sont disponibles et pertinentes au Québec. "
        "Donne des réponses détaillées, précises et bien structurées dans le contexte québécois. "
        "Si tu n'es pas certain d'une information, indique-le clairement.\n\n"
        "Documents pertinents (utilise si pertinents pour la question):\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create conversational RAG chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# Streamlit Configuration 
st.set_page_config(
    page_title="Chat avec Mistral 24b",
    page_icon="🍁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

custom_css = """
<style>
    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton>button {
        background-color: #0e1117;
        color: white;
    }
    .message-user {
        background-color: #004aad;
        padding: 10px;
        border-radius: 8px;
        color: white;
        margin-bottom: 10px;
    }
    .message-bot {
        background-color: #262730;
        padding: 10px;
        border-radius: 8px;
        color: white;
        margin-bottom: 10px;
    }
    .web-sources { 
        background-color: #1a1a2e;
        padding: 8px;
        border-radius: 6px;
        border-left: 3px solid #0066cc;
        margin-top: 10px;
    }
    .doc-sources {
        background-color: #1a1a2e;
        padding: 8px;
        border-radius: 6px;
        border-left: 3px solid #38d39f;
        margin-top: 10px;
    }
    .source-item { 
        margin-bottom: 8px;
        padding: 5px;
        background-color: #16213e;
        border-radius: 4px;
    }
    .source-title { 
        font-weight: bold;
        color: #66b3ff;
        margin-bottom: 4px;
    }
    .source-url { 
        color: #9db4d1;
        font-size: 0.9em;
        margin-bottom: 6px;
        word-break: break-all;
    }
    .source-snippet { 
        color: #d4d4d4;
        font-size: 0.9em;
        font-style: italic;
        line-height: 1.4;
    }
    .memory-indicator { 
        background-color: #2d1b69;
        padding: 6px;
        border-radius: 4px;
        border-left: 3px solid #8b5cf6;
        margin-top: 8px;
        font-size: 0.9em;
        color: #c4b5fd;
    }
    .docs-insufficient-indicator {
        background-color: #92400e;
        padding: 6px;
        border-radius: 4px;
        border-left: 3px solid #f59e0b;
        margin-top: 8px;
        font-size: 0.9em;
        color: #fcd34d;
    }
    .error-indicator {
        background-color: #7f1d1d;
        padding: 6px;
        border-radius: 4px;
        border-left: 3px solid #dc2626;
        margin-top: 8px;
        font-size: 0.9em;
        color: #fca5a5;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Session state for UI messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# --- Display title ---
st.markdown("<h1 style='text-align: center;'>🍁 Chat avec Mistral 24b</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Assistant répondant aux questions sur le système et les professions au Québec.</p>", unsafe_allow_html=True)

# Show system status
if not llm:
    st.error("⚠️ LLM non disponible - Vérifiez votre configuration")
elif not db:
    st.warning("⚠️ Base de données vectorielle non trouvée - Recherche documentaire indisponible")
elif not DDGS_AVAILABLE:
    st.warning("⚠️ Recherche web non disponible - Installez duckduckgo-search")

# --- Display chat history ---
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    web_sources = msg.get("web_sources", [])
    doc_sources = msg.get("doc_sources", [])
    used_memory = msg.get("used_memory", False)
    used_fallback = msg.get("used_fallback", False)
    docs_insufficient = msg.get("docs_insufficient", False)
    no_docs_found = msg.get("no_docs_found", False)
    conversation_based = msg.get("conversation_based", False)
    docs_not_relevant = msg.get("docs_not_relevant", False)
    no_sufficient_info = msg.get("no_sufficient_info", False)
    error = msg.get("error", False)

    block_class = "message-user" if role == "user" else "message-bot"
    st.markdown(f"<div class='{block_class}'><b>{'Toi' if role == 'user' else 'Mistral 24b'}:</b> {content}</div>", unsafe_allow_html=True)

    if role == "bot":
        # Show error indicator
        if error:
            st.markdown("<div class='error-indicator'><strong>❌ Erreur système</strong></div>", unsafe_allow_html=True)
        # Show memory indicator
        elif used_memory and conversation_based:
            st.markdown("<div class='memory-indicator'><strong>🧠 Réponse basée sur notre conversation</strong></div>", unsafe_allow_html=True)
        elif no_sufficient_info:
            st.markdown("<div class='docs-insufficient-indicator'><strong>❌ Informations insuffisantes trouvées</strong></div>", unsafe_allow_html=True)
        
        # Show web sources
        if web_sources:
            st.markdown("<div class='web-sources'><strong>🌐 Sources web:</strong></div>", unsafe_allow_html=True)
            for i, source in enumerate(web_sources, 1):
                title = source.get('title', 'Source Web')
                url = source.get('url', '')
                snippet = source.get('snippet', '')
                source_html = f"""
                <div class='source-item'>
                    <div class='source-title'>{i}. {title}</div>
                    <div class='source-url'>🔗 {url}</div>
                    <div class='source-snippet'>{snippet}</div>
                </div>
                """
                st.markdown(source_html, unsafe_allow_html=True)
        
        # Show document sources only if not memory/fallback and we have sources
        if doc_sources and not used_memory and not used_fallback:
            st.markdown("<div class='doc-sources'><strong>📄 Sources documentaires utilisées:</strong></div>", unsafe_allow_html=True)
            for i, doc_source in enumerate(doc_sources, 1):
                filename = doc_source.get('filename', 'inconnu')
                content_snippet = doc_source.get('content', '')
                source_html = f"""
                <div class='source-item'>
                    <div class='source-title'>{i}. {filename}</div>
                    <div class='source-snippet'>{content_snippet}</div>
                </div>
                """
                st.markdown(source_html, unsafe_allow_html=True)

# User input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ton message:", placeholder="Pose ta question", label_visibility="collapsed")
    submit_button = st.form_submit_button("Envoyer")

if submit_button and user_input:
    try:
        # Use the enhanced response system
        result = get_enhanced_response(user_input, None, st.session_state.session_id)
        
        answer = result["answer"]
        sources = result.get("context", [])
        used_web = result.get("used_web", False)
        web_sources = result.get("web_sources", [])
        used_fallback = result.get("used_fallback", False)
        used_memory = result.get("used_memory", False)
        doc_sources = result.get("doc_sources", [])
        docs_insufficient = result.get("docs_insufficient", False)
        no_docs_found = result.get("no_docs_found", False)
        conversation_based = result.get("conversation_based", False)
        docs_not_relevant = result.get("docs_not_relevant", False)
        no_sufficient_info = result.get("no_sufficient_info", False)
        error = result.get("error", False)

        # Update chat history only if LLM is available
        if llm:
            chat_history = get_session_history(st.session_state.session_id)
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(answer)

        # Clean response without any source information mixed in
        clean_response = answer

        # Save messages for UI
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Determine decision type for display
        if llm:
            chat_hist = get_session_history(st.session_state.session_id)
            decision = get_intelligent_decision(user_input, chat_hist)
            decision_type = decision.get("type", "documents")
        else:
            decision_type = "error"
        
        # Store the bot message with proper source separation
        bot_message = {
            "role": "bot", 
            "content": clean_response,
            "web_sources": web_sources if used_web else [],
            "doc_sources": doc_sources,
            "used_web": used_web,
            "used_memory": used_memory,
            "used_fallback": used_fallback,
            "decision_type": decision_type,
            "docs_insufficient": docs_insufficient,
            "no_docs_found": no_docs_found,
            "conversation_based": conversation_based,
            "docs_not_relevant": docs_not_relevant,
            "no_sufficient_info": no_sufficient_info,
            "error": error
        }

        # Add AI switched to web indicator
        if result.get("ai_switched_to_web", False):
            bot_message["ai_switched_to_web"] = True

        print(f"Bot message doc_sources: {len(doc_sources)} sources")
        print(f"Used web: {used_web}, Used memory: {used_memory}")
        print(f"Docs insufficient: {docs_insufficient}, No docs found: {no_docs_found}")

        st.session_state.messages.append(bot_message)

    except Exception as e:
        st.error(f"Erreur lors du traitement de votre demande: {str(e)}")
        print(f"Erreur détaillée: {e}")

    # Reload to show new messages
    st.rerun()