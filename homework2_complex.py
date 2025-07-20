#!/usr/bin/env python3
"""
Homework 2: Unsupervised Clustering Analysis
============================================

This script implements a comprehensive text clustering pipeline comparing TF-IDF 
and sentence embedding vectorisation methods on news article titles. The analysis
evaluates clustering performance using multiple metrics to determine which 
vectorisation approach yields superior topic identification.

Dependencies:
    - pandas: Data manipulation and analysis
    - numpy: Numerical computing
    - scikit-learn: Machine learning algorithms and metrics
    - sentence-transformers: Pre-trained sentence embeddings
    - matplotlib/seaborn: Data visualisation
    - gdown: Google Drive file downloading
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os
import warnings
warnings.filterwarnings('ignore')

class NewsClusteringAnalysis:
    """
    A comprehensive clustering analysis class that compares TF-IDF and sentence 
    embedding approaches for news article classification.
    
    This class encapsulates the entire pipeline from data loading through 
    performance evaluation, providing a systematic comparison of vectorisation
    methods for unsupervised document clustering.
    """
    
    def __init__(self, random_state=42):
        """
        Initialise the clustering analysis with reproducible random state.
        
        Args:
            random_state (int): Seed for reproducible results across runs
        """
        self.random_state = random_state
        self.data = None
        self.true_labels = None
        self.unique_topics = None
        self.n_clusters = None
        
        # Vectorisation results storage
        self.tfidf_vectors = None
        self.embedding_vectors = None
        
        # Clustering results storage
        self.tfidf_clusters = None
        self.embedding_clusters = None
        
        # Performance metrics storage
        self.results = {}
        
        print("NewsClusteringAnalysis initialised successfully")
    
    def load_dataset(self):
        """
        Download and load the news dataset from Google Drive.
        
        The dataset contains news articles with titles and associated topics,
        providing ground truth labels for clustering evaluation.
        
        Returns:
            pd.DataFrame: Loaded dataset with 'title' and 'topic' columns
        """
        print("Downloading dataset from Google Drive...")
        
        # Google Drive file ID extracted from the provided URL
        file_id = "1ADBC1whqSJZ4FPuBbhhWjWOmFI1XW1Kb"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            # Attempt direct download using gdown
            output_path = "news_AP.csv"
            gdown.download(url, output_path, quiet=False)
            
            # Load the CSV file
            self.data = pd.read_csv(output_path)
            print(f"Dataset loaded successfully: {self.data.shape[0]} articles")
            
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please manually download news_AP.csv and place it in the current directory")
            
            # Try to load from current directory if download fails
            if os.path.exists("news_AP.csv"):
                self.data = pd.read_csv("news_AP.csv")
                print(f"Dataset loaded from local file: {self.data.shape[0]} articles")
            else:
                raise FileNotFoundError("news_AP.csv not found. Please download manually.")
        
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the dataset for clustering analysis.
        
        This method cleans the data, extracts unique topics, and prepares
        ground truth labels for performance evaluation.
        """
        print("Preprocessing dataset...")
        
        # Remove any rows with missing titles or topics
        self.data = self.data.dropna(subset=['title', 'topic'])
        
        # Extract unique topics and determine number of clusters
        self.unique_topics = sorted(self.data['topic'].unique())
        self.n_clusters = len(self.unique_topics)
        
        # Create numerical labels for ground truth evaluation
        topic_to_label = {topic: idx for idx, topic in enumerate(self.unique_topics)}
        self.true_labels = self.data['topic'].map(topic_to_label).values
        
        print(f"Found {self.n_clusters} unique topics: {self.unique_topics}")
        print(f"Dataset contains {len(self.data)} articles after preprocessing")
        
        # Display topic distribution
        topic_counts = self.data['topic'].value_counts()
        print("\nTopic distribution:")
        for topic, count in topic_counts.items():
            print(f"  {topic}: {count} articles ({count/len(self.data)*100:.1f}%)")
    
    def vectorise_tfidf(self, max_features=5000, min_df=2, max_df=0.95):
        """
        Vectorise document titles using TF-IDF (Term Frequency-Inverse Document Frequency).
        
        TF-IDF creates sparse vectors where each dimension represents a term's
        importance within a document relative to the entire corpus.
        
        Args:
            max_features (int): Maximum number of features to extract
            min_df (int): Minimum document frequency for terms
            max_df (float): Maximum document frequency ratio for terms
            
        Returns:
            scipy.sparse matrix: TF-IDF vectors for all documents
        """
        print("Vectorising documents using TF-IDF...")
        
        # Initialise TF-IDF vectoriser with preprocessing parameters
        vectoriser = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',  # Remove common English stop words
            lowercase=True,        # Normalise to lowercase
            strip_accents='ascii'  # Remove accents for consistency
        )
        
        # Transform document titles to TF-IDF vectors
        self.tfidf_vectors = vectoriser.fit_transform(self.data['title'])
        
        print(f"TF-IDF vectorisation complete: {self.tfidf_vectors.shape} (documents × features)")
        print(f"Vocabulary size: {len(vectoriser.vocabulary_)}")
        print(f"Sparsity: {(1 - self.tfidf_vectors.nnz / np.prod(self.tfidf_vectors.shape))*100:.1f}%")
        
        return self.tfidf_vectors
    
    def vectorise_embeddings(self, model_name='all-MiniLM-L6-v2'):
        """
        Vectorise document titles using pre-trained sentence embeddings.
        
        Sentence embeddings create dense vectors that capture semantic meaning
        through deep learning models trained on large text corpora.
        
        Args:
            model_name (str): Name of the pre-trained sentence transformer model
            
        Returns:
            numpy.ndarray: Dense embedding vectors for all documents
        """
        print(f"Vectorising documents using sentence embeddings ({model_name})...")
        
        # Load pre-trained sentence transformer model
        model = SentenceTransformer(model_name)
        
        # Generate embeddings for all document titles
        titles = self.data['title'].tolist()
        self.embedding_vectors = model.encode(titles, show_progress_bar=True)
        
        print(f"Embedding vectorisation complete: {self.embedding_vectors.shape} (documents × dimensions)")
        print(f"Embedding dimension: {self.embedding_vectors.shape[1]}")
        
        return self.embedding_vectors
    
    def perform_clustering(self, n_init=10, max_iter=300):
        """
        Perform K-means clustering on both vectorisation methods.
        
        K-means clustering partitions documents into k clusters by minimising
        within-cluster sum of squared distances to centroids.
        
        Args:
            n_init (int): Number of random initialisations
            max_iter (int): Maximum number of iterations
        """
        print(f"Performing K-means clustering with k={self.n_clusters}...")
        
        # Cluster TF-IDF vectors
        print("Clustering TF-IDF vectors...")
        tfidf_kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=n_init,
            max_iter=max_iter
        )
        self.tfidf_clusters = tfidf_kmeans.fit_predict(self.tfidf_vectors)
        
        # Cluster embedding vectors
        print("Clustering embedding vectors...")
        embedding_kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=n_init,
            max_iter=max_iter
        )
        self.embedding_clusters = embedding_kmeans.fit_predict(self.embedding_vectors)
        
        print("Clustering completed for both methods")
    
    def calculate_metrics(self, predicted_labels, method_name):
        """
        Calculate comprehensive performance metrics for clustering results.
        
        This method computes precision, recall, F1-score, and accuracy by
        finding the optimal mapping between predicted clusters and true topics.
        
        Args:
            predicted_labels (numpy.ndarray): Cluster assignments
            method_name (str): Name of the vectorisation method
            
        Returns:
            dict: Dictionary containing all performance metrics
        """
        print(f"Calculating metrics for {method_name}...")
        
        # Create confusion matrix
        cm = confusion_matrix(self.true_labels, predicted_labels)
        
        # Find optimal mapping between clusters and topics using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        cost_matrix = -cm  # Negative because we want to maximise overlap
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Apply optimal mapping to predicted labels
        mapping = dict(zip(col_indices, row_indices))
        mapped_predictions = np.array([mapping.get(label, label) for label in predicted_labels])
        
        # Calculate metrics with optimal mapping
        accuracy = accuracy_score(self.true_labels, mapped_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, mapped_predictions, average=None, zero_division=0
        )
        
        # Calculate macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Store detailed results
        metrics = {
            'confusion_matrix': confusion_matrix(self.true_labels, mapped_predictions),
            'accuracy': accuracy,
            'precision_per_topic': precision,
            'recall_per_topic': recall,
            'f1_per_topic': f1,
            'support_per_topic': support,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'topics': self.unique_topics
        }
        
        return metrics
    
    def evaluate_performance(self):
        """
        Evaluate clustering performance for both vectorisation methods.
        
        This method calculates and stores comprehensive metrics for comparison
        between TF-IDF and embedding-based clustering approaches.
        """
        print("Evaluating clustering performance...")
        
        # Calculate metrics for both methods
        self.results['tfidf'] = self.calculate_metrics(self.tfidf_clusters, "TF-IDF")
        self.results['embeddings'] = self.calculate_metrics(self.embedding_clusters, "Embeddings")
        
        print("Performance evaluation completed")
    
    def display_results(self):
        """
        Display comprehensive clustering results and performance comparison.
        
        This method presents results in a clear, structured format suitable
        for academic reporting and analysis interpretation.
        """
        print("\n" + "="*80)
        print("CLUSTERING PERFORMANCE RESULTS")
        print("="*80)
        
        methods = ['tfidf', 'embeddings']
        method_names = ['TF-IDF', 'Sentence Embeddings']
        
        for method, method_name in zip(methods, method_names):
            results = self.results[method]
            
            print(f"\n{method_name.upper()} RESULTS:")
            print("-" * 50)
            print(f"Overall Accuracy: {results['accuracy']:.3f}")
            print(f"Macro Precision:  {results['macro_precision']:.3f}")
            print(f"Macro Recall:     {results['macro_recall']:.3f}")
            print(f"Macro F1-Score:   {results['macro_f1']:.3f}")
            
            print(f"\nPer-Topic Performance:")
            print(f"{'Topic':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 70)
            
            for i, topic in enumerate(results['topics']):
                precision = results['precision_per_topic'][i]
                recall = results['recall_per_topic'][i]
                f1 = results['f1_per_topic'][i]
                support = results['support_per_topic'][i]
                
                print(f"{topic:<20} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")
        
        # Comparison summary
        print(f"\n{'PERFORMANCE COMPARISON'}")
        print("="*50)
        
        tfidf_acc = self.results['tfidf']['accuracy']
        emb_acc = self.results['embeddings']['accuracy']
        
        if emb_acc > tfidf_acc:
            better_method = "Sentence Embeddings"
            improvement = (emb_acc - tfidf_acc) / tfidf_acc * 100
        else:
            better_method = "TF-IDF"
            improvement = (tfidf_acc - emb_acc) / emb_acc * 100
        
        print(f"Superior Method: {better_method}")
        print(f"Accuracy Improvement: {improvement:.1f}%")
        
        print(f"\nTF-IDF Accuracy:           {tfidf_acc:.3f}")
        print(f"Embedding Accuracy:        {emb_acc:.3f}")
        print(f"Absolute Difference:       {abs(emb_acc - tfidf_acc):.3f}")
    
    def create_visualisations(self):
        """
        Generate comprehensive visualisations for clustering analysis.
        
        Creates confusion matrices and performance comparison charts
        to facilitate visual interpretation of results.
        """
        print("Creating visualisations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('News Article Clustering Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot confusion matrices
        methods = ['tfidf', 'embeddings']
        method_names = ['TF-IDF', 'Sentence Embeddings']
        
        for i, (method, method_name) in enumerate(zip(methods, method_names)):
            cm = self.results[method]['confusion_matrix']
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.unique_topics,
                yticklabels=self.unique_topics,
                ax=axes[i, 0]
            )
            axes[i, 0].set_title(f'{method_name} Confusion Matrix')
            axes[i, 0].set_xlabel('Predicted Topic')
            axes[i, 0].set_ylabel('True Topic')
        
        # Performance comparison bar chart
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        tfidf_scores = [self.results['tfidf'][metric] for metric in metrics]
        embedding_scores = [self.results['embeddings'][metric] for metric in metrics]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, tfidf_scores, width, label='TF-IDF', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, embedding_scores, width, label='Embeddings', alpha=0.8, color='lightcoral')
        
        axes[0, 1].set_title('Performance Metrics Comparison')
        axes[0, 1].set_xlabel('Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metric_names)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (tfidf_score, emb_score) in enumerate(zip(tfidf_scores, embedding_scores)):
            axes[0, 1].text(i - width/2, tfidf_score + 0.01, f'{tfidf_score:.3f}', 
                           ha='center', va='bottom', fontsize=9)
            axes[0, 1].text(i + width/2, emb_score + 0.01, f'{emb_score:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        # Per-topic F1 scores comparison
        topics = self.unique_topics
        tfidf_f1 = self.results['tfidf']['f1_per_topic']
        embedding_f1 = self.results['embeddings']['f1_per_topic']
        
        x_topics = np.arange(len(topics))
        
        axes[1, 1].bar(x_topics - width/2, tfidf_f1, width, label='TF-IDF', alpha=0.8, color='skyblue')
        axes[1, 1].bar(x_topics + width/2, embedding_f1, width, label='Embeddings', alpha=0.8, color='lightcoral')
        
        axes[1, 1].set_title('F1-Score by Topic')
        axes[1, 1].set_xlabel('Topics')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_xticks(x_topics)
        axes[1, 1].set_xticklabels(topics, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('clustering_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualisations saved as 'clustering_analysis_results.png'")
    
    def run_complete_analysis(self):
        """
        Execute the complete clustering analysis pipeline.
        
        This method orchestrates the entire analysis from data loading
        through results presentation, providing a comprehensive comparison
        of vectorisation methods for document clustering.
        """
        print("Starting complete clustering analysis pipeline...")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        self.load_dataset()
        self.preprocess_data()
        
        # Step 2: Vectorise documents using both methods
        self.vectorise_tfidf()
        self.vectorise_embeddings()
        
        # Step 3: Perform clustering
        self.perform_clustering()
        
        # Step 4: Evaluate performance
        self.evaluate_performance()
        
        # Step 5: Display results
        self.display_results()
        
        # Step 6: Create visualisations
        self.create_visualisations()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Files generated:")
        print("- clustering_analysis_results.png: Comprehensive visualisations")
        print("- news_AP.csv: Original dataset (if downloaded)")
        
        return self.results

def main():
    """
    Main execution function for the clustering analysis.
    
    This function initialises the analysis class and executes the complete
    pipeline, providing a robust comparison of TF-IDF and embedding-based
    clustering for news article topic identification.
    """
    print("News Article Clustering Analysis")
    print("================================")
    print("Comparing TF-IDF vs. Sentence Embeddings for Document Clustering")
    print()
    
    # Initialise and run analysis
    analyser = NewsClusteringAnalysis(random_state=42)
    results = analyser.run_complete_analysis()
    
    print("\nAnalysis completed successfully!")
    print("Review the generated report and visualisations for detailed insights.")
    
    return results

if __name__ == "__main__":
    # Execute the complete analysis
    results = main()