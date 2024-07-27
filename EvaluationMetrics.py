# EvaluationMetrics.py

import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Define BERT tokenizer and model for later use
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def convert_to_list(data):
    # Convert nested list of strings into a flat list of strings
    data_list = []
    for listItem in data:
        for item in listItem.split():
            data_list.append(item)
    return data_list

def precision_recall(retrieved, ground_truth):
    
    print('\n\n\n\n Precision recall function: \n')

    print('\n\nprecision_recall - retrieved:\n', retrieved)
    retrieved_set = set(convert_to_list(retrieved))# Convert lists to sets of strings
    print('\n\n retrieved set: \n', retrieved_set)

    print('\n\n precision_recall - ground truth: \n', ground_truth)
    ground_truth_set = set(convert_to_list(ground_truth))
    print('\n\n ground_truth_set: \n', ground_truth_set)
    
    
    # Calculate precision and recall
    true_positives = retrieved_set.intersection(ground_truth_set)
    print('\n\n true_positives in precision recall function: \n', len(true_positives))
    print('\n retrieved_set length: \n', len(retrieved_set))
    print('\n ground_truth_set length: \n', len(ground_truth_set))
    
    # Context Precision: Measure how accurately the retrieved context matches the user's query
    precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0
    
    # Context Recall: Evaluate the ability to retrieve all relevant contexts for the user's query
    recall = len(true_positives) / len(ground_truth_set) if ground_truth_set else 0

    return precision, recall


def relevance_score(retrieved_context, user_query):
    # Context Relevance: Assess the relevance of the retrieved context to the user's query
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([retrieved_context, user_query])
    similarity_matrix = cosine_similarity(vectors)
    relevance = similarity_matrix[0, 1]
    return relevance


def extract_custom_entities(text):
    doc = nlp(text)
    entities = {"item": None, "color": None, "target_audience": None}
    
    # Use SpaCy's NER for initial extraction
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "WORK_OF_ART", "EVENT"]:
            entities["item"] = ent.text.lower()
        elif ent.label_ == "COLOR":  # SpaCy doesn't have COLOR by default
            entities["color"] = ent.text.lower()
        elif ent.label_ in ["PERSON", "NORP", "ORG"]:  # NORP covers groups like 'boys'
            entities["target_audience"] = ent.text.lower()

    # Custom extraction logic for our specific entities
    for token in doc:
        if token.text.lower() in ["sandals", "shoes", "sneakers", "boots","sweatshirt"]:  # Add more items as needed
            entities["item"] = token.text.lower()
        elif token.text.lower() in ["purple", "blue", "red", "green", "yellow"]:  # Add more colors as needed
            entities["color"] = token.text.lower()
        elif token.text.lower() in ["boys", "girls", "men", "women", "kids", "adults"]:
            entities["target_audience"] = token.text.lower()

    # Filter out None values
    entities = {k: v for k, v in entities.items() if v}
    
    return entities


def entity_recall(retrieved_context, user_query):
    # Context Entity Recall: Determine the ability to recall relevant entities within the context
    retrieved_entities = set(extract_custom_entities(retrieved_context).values())
    query_entities = set(extract_custom_entities(user_query).values())
    # retrieved_entities = set([ent.text for ent in nlp(retrieved_context).ents])
    # query_entities = set([ent.text for ent in nlp(user_query).ents])
    print('\n\n\n retrieved entities set: ', retrieved_entities)
    print('\n\n\n query entities set: ', query_entities)
    true_positives = len(retrieved_entities & query_entities)
    print('\n\n\n true_positives entities recall: ', true_positives)
    recall = true_positives / len(query_entities) if len(query_entities) > 0 else 0
    
    return recall

def noise_robustness(query, noise_level=0.1):
    # Convert input to list
    query = ' '.join(query)
    
    # Noise Robustness: Test the system's ability to handle noisy or irrelevant inputs
    noisy_query = ''.join([char if np.random.rand() > noise_level else '' for char in query])
    return noisy_query

def check_faithfulness(generated_answer, ground_truth):
    # Faithfulness: Measure the accuracy and reliability of the generated answers
    inputs = tokenizer(generated_answer, ground_truth, return_tensors='pt')
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    faithfulness = torch.sigmoid(logits).mean().item()  # Get the mean value for faithfulness
    
    return faithfulness

def bleu_score(generated_answer, ground_truth):
    # Answer Relevance: Evaluate the relevance of the generated answers to the user's query using BLEU score
    reference = [ground_truth.split()]
    candidate = generated_answer.split()
    score = sentence_bleu(reference, candidate)
    return score

def rouge_score(generated_answer, ground_truth):
    # Information Integration: Assess the ability to integrate and present information cohesively using ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(generated_answer, ground_truth)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

def counterfactual_robustness(generated_answer, counterfactual_answer):
    # Convert inputs to list
    generated_answer = ' '.join(generated_answer)
    counterfactual_answer = ' '.join(counterfactual_answer)
    
    # Counterfactual Robustness: Test the robustness of the system against counterfactual or contradictory queries
    return generated_answer != counterfactual_answer

def negative_rejection(generated_answer):
    # Convert input to list
    generated_answer = ' '.join(convert_to_list([generated_answer]))
    
    # Negative Rejection: Measure the system's ability to reject and handle negative or inappropriate queries
    negative_keywords = ['no', 'not', 'none', 'nothing', 'never', 'out of']
    return any(negative in generated_answer.lower() for negative in negative_keywords)

def measure_latency(query, rag_pipeline):
    # Latency: Measure the response time of the system from receiving a query to delivering an answer
    start_time = time.time()
    response = rag_pipeline(query)
    end_time = time.time()
    latency = end_time - start_time
    return latency

def evaluate_metrics(queries, ground_truths, get_response, vector_store):
    metrics = {
        'precision': [],
        'recall': [],
        'relevance': [],
        'entity_recall': [],
        'faithfulness': [],
        'bleu': [],
        'rouge1': [],
        'rougeL': [],
        'latency': [],
        'noise_robustness': [],
        'counterfactual_robustness': [],
        'negative_rejection': []
    }
    
    for query, ground_truth in zip(queries, ground_truths):
        # Perform retrieval and generation
        retrieved_contexts = [doc.page_content for doc in vector_store.search(query, search_type='similarity')]
        print("\n\n\n\n**********Retrieved Contexts from vector store: ************\n", retrieved_contexts)
        generated_answer = get_response(query)
        print("\n\n\n\n**********Generated Answer: ************\n", generated_answer)

        # Calculate retrieval metrics
        precision, recall = precision_recall(retrieved_contexts, ground_truth['contexts'])
        relevance = relevance_score(' '.join(retrieved_contexts), query)
        entity_recall_score = entity_recall(' '.join(convert_to_list(retrieved_contexts)), query)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['relevance'].append(relevance)
        metrics['entity_recall'].append(entity_recall_score)
        
        # Calculate noise robustness
        noisy_query = noise_robustness(query)
        noisy_generated_answer = get_response(noisy_query)
        noise_robustness_score = relevance_score(noisy_generated_answer, query)
        metrics['noise_robustness'].append(noise_robustness_score)

        # Calculate generation metrics
        faithfulness = check_faithfulness(generated_answer, ground_truth['answer'])
        bleu = bleu_score(generated_answer, ground_truth['answer'])
        rouge1, rougeL = rouge_score(generated_answer, ground_truth['answer'])
        
        metrics['faithfulness'].append(faithfulness)
        metrics['bleu'].append(bleu)
        metrics['rouge1'].append(rouge1)
        metrics['rougeL'].append(rougeL)
        
        # Calculate counterfactual robustness
        counterfactual_query = 'not ' + query
        counterfactual_generated_answer = get_response(counterfactual_query)
        counterfactual_robustness_score = counterfactual_robustness(generated_answer, counterfactual_generated_answer)
        metrics['counterfactual_robustness'].append(counterfactual_robustness_score)
        
        # Calculate negative rejection
        negative_rejection_score = negative_rejection(generated_answer)
        metrics['negative_rejection'].append(negative_rejection_score)
    
        # Calculate latency
        latency = measure_latency(query, get_response)
        metrics['latency'].append(latency)
    return metrics


