//! Seeder & Swarm — BitTorrent-style piece distribution and retrieval.
//!
//! Each Seeder holds a shard of pieces and computes relevance in parallel.
//! The Swarm is all seeders together, queried via broadcast-gather.
//!
//! BitTorrent mapping:
//!   Seeder        → Worker holding pieces in memory
//!   Swarm         → All workers on the machine
//!   Query         → Client requesting pieces
//!   _dot score    → Relevance (replaces rarest-first with relevance-first)
//!   Top-K results → Best pieces assembled into context

use crate::k::K;
use crate::piece::PieceManager;
use crate::va;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

/// Command sent to worker threads
pub enum SeederCommand {
    Query(K),
}

/// Result received from worker threads
pub enum SeederResult {
    QueryResult(Vec<QueryResult>),
}

/// A worker thread that holds a shard of pieces and processes queries.
///
/// This replaces the passive "Seeder" struct.
/// Now, a Seeder is an active thread.
pub struct SeederThread {
    /// Channel to send queries TO this worker
    pub tx: Sender<SeederCommand>,
}

/// A query result — one piece with its relevance score.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Piece ID
    pub piece_id: usize,
    /// Relevance score (from _dot)
    pub score: f64,
    /// Source file
    pub source: String,
    /// Content preview (first N chars)
    pub preview: String,
    /// Full content
    #[allow(dead_code)]
    pub content: String,
}

/// The Swarm — manages the active seeder threads.
pub struct Swarm {
    /// Active worker threads
    seeders: Vec<SeederThread>,
    /// Channel to receive results FROM workers
    result_rx: Receiver<SeederResult>,
    /// Number of workers
    num_threads: usize,
    /// Vocabulary size (embedding dimension)
    vocab_size: usize,
}

impl Swarm {
    /// Build a swarm from a PieceManager, sharding data across cores.
    pub fn from_pieces(manager: &PieceManager) -> Swarm {
        // 1. Detect cores (default to 4 if detection fails, or use all logical cores)
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        println!("   [Swarm] Spawning {} active seeder threads...", num_threads);

        // 2. Partition pieces (deduplicated)
        let mut unique_pieces = Vec::new();
        let mut seen_hashes = std::collections::HashSet::new();
        for piece in &manager.pieces {
            if !seen_hashes.contains(&piece.hash) {
                seen_hashes.insert(piece.hash);
                unique_pieces.push(piece.clone());
            }
        }

        let total_pieces = unique_pieces.len();
        let chunk_size = (total_pieces + num_threads - 1) / num_threads; // Ceiling division

        // 3. Create channels for results (Many-to-One)
        let (result_tx, result_rx) = channel();

        let mut seeders = Vec::with_capacity(num_threads);

        // 4. Spawn threads
        for i in 0..num_threads {
            // Take a slice of pieces for this thread
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, total_pieces);
            
            let shard = if start < total_pieces {
                unique_pieces[start..end].to_vec()
            } else {
                Vec::new() // Threads with no work just stay idle
            };

            // Channel for sending queries TO this thread
            let (tx, rx) = channel::<SeederCommand>();
            
            // Clone result sender for this thread
            let my_result_tx = result_tx.clone();
            
            let _handle = thread::spawn(move || {
                // Thread Loop: Wait for queries
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        SeederCommand::Query(query) => {
                            // "Active Seeder" Logic: Scan my shard
                            let mut local_results = Vec::with_capacity(shard.len());
                            
                            for piece in &shard {
                                // Compute score (FMA / Dot Product)
                                let dot_res = va::dot(&query, &piece.embedding);
                                let score = match dot_res.data {
                                    crate::k::KData::Floats(v) => v[0],
                                    crate::k::KData::Ints(v) => v[0] as f64,
                                    _ => 0.0,
                                };

                                if score > 0.001 { // Optimization: Don't send zero-score noise
                                     local_results.push(QueryResult {
                                        piece_id: piece.id,
                                        score,
                                        source: piece.source.display().to_string(),
                                        preview: piece.content.chars().take(100).collect::<String>(),
                                        content: piece.content.clone(),
                                    });
                                }
                            }
                            // Send my local results back to main thread
                            let _ = my_result_tx.send(SeederResult::QueryResult(local_results));
                        },
                    }
                }
            });

            seeders.push(SeederThread {
                tx,
            });
        }

        Swarm {
            seeders,
            result_rx,
            num_threads,
            vocab_size: manager.vocab_size(),
        }
    }

    /// Parallel Query: Broadcast to all threads, gather results.
    pub fn query(&self, query_embedding: &K, top_k: usize) -> Vec<QueryResult> {
        // 1. Broadcast query to all workers
        for seeder in &self.seeders {
            let _ = seeder.tx.send(SeederCommand::Query(query_embedding.clone()));
        }

        // 2. Gather results from all workers
        let mut all_results = Vec::new();
        for _ in 0..self.num_threads {
            match self.result_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(result) => {
                    match result {
                        SeederResult::QueryResult(mut shard_results) => {
                             all_results.append(&mut shard_results);
                        },
                    }
                }
                Err(e) => {
                    eprintln!("[Swarm] Warning: gather timed out or worker thread panicked: {:?}", e);
                    break;
                }
            }
        }

        // 3. Sort and truncate (Main thread reduction)
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(top_k);
        
        all_results
    }

    /// How many seeders in the swarm?
    pub fn seeder_count(&self) -> usize {
        self.seeders.len()
    }
    pub fn embedding_dim(&self) -> usize {
        self.vocab_size
    }
}
