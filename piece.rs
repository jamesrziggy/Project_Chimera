//! BitTorrent piece definitions for Project Chimera.

use crate::k::K;
use std::path::PathBuf;

/// A piece represents a chunk of data with its embedding, hash, and source metadata.
#[derive(Debug, Clone)]
pub struct Piece {
    /// Unique piece index
    pub id: usize,
    /// BitTorrent-style piece hash (SHA-1 20-byte hash or similar)
    pub hash: [u8; 20],
    /// Semantic embedding of the piece content
    pub embedding: K,
    /// Path to the source file this piece belongs to
    pub source: PathBuf,
    /// The text content of the piece
    pub content: String,
}

/// Manages a collection of pieces.
#[derive(Debug, Clone)]
pub struct PieceManager {
    /// List of pieces managed
    pub pieces: Vec<Piece>,
    /// Vocabulary/embedding dimension size
    pub vocab: usize,
}

impl PieceManager {
    /// Create a new PieceManager
    pub fn new(pieces: Vec<Piece>, vocab: usize) -> Self {
        Self { pieces, vocab }
    }

    /// Returns the vocabulary size (embedding dimension)
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }
}
