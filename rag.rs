//! Retrieval-Augmented Generation (RAG) components for Project Chimera.
//! Contains DocumentDatabase, TF-IDF vectorizer, and Hermes prompt template formatter.

use std::collections::{HashMap, HashSet};
use crate::k::{K, KData};
use crate::va;

/// Clean and tokenize input text.
/// Converts to lowercase, replaces non-alphanumeric characters with spaces,
/// and splits by whitespace.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Representation of a single document in the database.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub id: usize,
    pub content: String,
}

/// A simple document database that maps documents to their dense K vector representations.
pub struct DocumentDatabase {
    pub documents: Vec<Document>,
    pub vectors: Vec<K>,
}

/// A term frequency-inverse document frequency vectorizer.
pub struct TfidfVectorizer {
    pub vocabulary: HashMap<String, usize>,
    pub idfs: Vec<f64>,
}

impl TfidfVectorizer {
    /// Create a new, un-fitted TF-IDF vectorizer.
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            idfs: Vec::new(),
        }
    }

    /// Fit the vectorizer on a corpus of documents, building the vocabulary and calculating IDFs.
    pub fn fit(&mut self, docs: &[String]) {
        let mut df_counts: HashMap<String, usize> = HashMap::new();

        // Count document frequency (DF) for each term
        for doc in docs {
            let tokens = tokenize(doc);
            let unique_tokens: HashSet<String> = tokens.into_iter().collect();
            for token in unique_tokens {
                *df_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Sort terms for deterministic vocabulary indexing
        let mut sorted_terms: Vec<String> = df_counts.keys().cloned().collect();
        sorted_terms.sort();

        let mut vocabulary = HashMap::new();
        let mut idfs = Vec::new();
        let n_docs = docs.len();

        for (idx, term) in sorted_terms.into_iter().enumerate() {
            vocabulary.insert(term.clone(), idx);
            let df = df_counts[&term];
            // Standard smooth IDF: ln( (1 + N) / (1 + DF) ) + 1.0
            let idf = ((1.0 + n_docs as f64) / (1.0 + df as f64)).ln() + 1.0;
            idfs.push(idf);
        }

        self.vocabulary = vocabulary;
        self.idfs = idfs;
    }

    /// Transform a text document into a L2-normalized TF-IDF vector K object.
    pub fn transform(&self, text: &str) -> K {
        if self.vocabulary.is_empty() {
            return K::from_floats(Vec::new());
        }

        let tokens = tokenize(text);
        let mut term_counts = HashMap::new();
        for token in tokens {
            if self.vocabulary.contains_key(&token) {
                *term_counts.entry(token).or_insert(0.0) += 1.0;
            }
        }

        let mut vector = vec![0.0; self.vocabulary.len()];
        for (term, count) in term_counts {
            if let Some(&idx) = self.vocabulary.get(&term) {
                vector[idx] = count * self.idfs[idx];
            }
        }

        // L2 normalization
        let sum_sq: f64 = vector.iter().map(|&x| x * x).sum();
        let norm = sum_sq.sqrt();
        if norm > 0.0 {
            for val in &mut vector {
                *val /= norm;
            }
        }

        K::from_floats(vector)
    }
}

/// The RAG retrieval pipeline combining the TF-IDF vectorizer and document database.
pub struct RetrievalPipeline {
    pub vectorizer: TfidfVectorizer,
    pub db: DocumentDatabase,
}

impl RetrievalPipeline {
    /// Create a new retrieval pipeline and index the provided documents.
    pub fn new(docs: Vec<Document>) -> Self {
        let doc_contents: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
        let mut vectorizer = TfidfVectorizer::new();
        vectorizer.fit(&doc_contents);

        let mut vectors = Vec::new();
        for doc in &docs {
            let vec_k = vectorizer.transform(&doc.content);
            vectors.push(vec_k);
        }

        let db = DocumentDatabase {
            documents: docs,
            vectors,
        };

        Self { vectorizer, db }
    }

    /// Retrieve the top K matching documents for a query based on cosine similarity.
    pub fn retrieve(&self, query: &str, k: usize) -> Vec<(Document, f64)> {
        if self.db.documents.is_empty() {
            return Vec::new();
        }

        let query_vec = self.vectorizer.transform(query);
        let mut scores = Vec::new();

        for (i, doc_vec) in self.db.vectors.iter().enumerate() {
            let score = if query_vec.n == 0 || doc_vec.n == 0 {
                0.0
            } else {
                // Compute dot product of two normalized vectors to get cosine similarity
                let dot_k = va::dot(&query_vec, doc_vec);
                match dot_k.data {
                    KData::Floats(ref v) if !v.is_empty() => v[0],
                    KData::Ints(ref v) if !v.is_empty() => v[0] as f64,
                    _ => 0.0,
                }
            };
            scores.push((self.db.documents[i].clone(), score));
        }

        // Sort by score descending, then by document ID ascending for deterministic output
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.id.cmp(&b.0.id))
        });

        scores.truncate(k);
        scores
    }
}

/// Format the retrieval prompt conforming to the Hermes 2 / ChatML specifications.
pub fn format_hermes_prompt(system_prompt: &str, context: &str, query: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(system_prompt);
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>user\n");
    if !context.is_empty() {
        prompt.push_str("Context:\n");
        prompt.push_str(context);
        prompt.push_str("\n\n");
    }
    prompt.push_str("Question: ");
    prompt.push_str(query);
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenization() {
        let text = "Rust's safety, speed, and concurrency.";
        let tokens = tokenize(text);
        assert_eq!(tokens, vec!["rust", "s", "safety", "speed", "and", "concurrency"]);
    }

    #[test]
    fn test_tfidf_vectorizer() {
        let mut vectorizer = TfidfVectorizer::new();
        let docs = vec![
            "Rust is safe".to_string(),
            "Python is simple".to_string(),
        ];
        vectorizer.fit(&docs);

        // Vocab should contain "is", "python", "rust", "safe", "simple"
        assert!(vectorizer.vocabulary.contains_key("is"));
        assert!(vectorizer.vocabulary.contains_key("rust"));
        assert!(vectorizer.vocabulary.contains_key("python"));
        assert!(vectorizer.vocabulary.contains_key("safe"));
        assert!(vectorizer.vocabulary.contains_key("simple"));

        // Transform a document
        let vec_k = vectorizer.transform("Rust is safe");
        assert_eq!(vec_k.t, 2); // FloatArray
        assert_eq!(vec_k.n, 5); // 5 elements in vocab

        // Verify it is L2 normalized
        let data = vec_k.kf_data();
        let sum_sq: f64 = data.iter().map(|x| x * x).sum();
        assert!((sum_sq - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_retrieval_pipeline() {
        let docs = vec![
            Document { id: 1, content: "Rust concurrency and safety".to_string() },
            Document { id: 2, content: "Python data science and AI".to_string() },
        ];
        let pipeline = RetrievalPipeline::new(docs);
        let matches = pipeline.retrieve("Rust safety", 1);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0.id, 1);
        assert!(matches[0].1 > 0.0);
    }

    #[test]
    fn test_hermes_prompt_formatter() {
        let prompt = format_hermes_prompt("system", "context", "query");
        let expected = "<|im_start|>system\nsystem<|im_end|>\n<|im_start|>user\nContext:\ncontext\n\nQuestion: query<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(prompt, expected);
    }
}

