//! # flyrag
//!
//! Retrieval that **learns from verified usefulness**, built on
//! [`kenyon`]'s mushroom-body codes.
//!
//! A query and every document become sparse Kenyon-cell codes; candidates
//! are the documents whose codes overlap the query's. Each document also has
//! a learned readout: two banks of weights over KC units, "use" and "skip".
//! When the user verifies that an offered document helped (or misled), one
//! bank is depressed on the query's active units, so the document rises (or
//! falls) for queries *like this one*, and for no others. Updates are bounded
//! and driven by prediction error (an outcome that was already expected
//! teaches nothing); only documents that were offered can learn; nothing is
//! ever overwritten, and every weight decays back toward its prior. A fly
//! Bloom filter flags queries unlike anything seen before.
//!
//! What it is for: a retrieval layer that should get better from outcomes
//! and say when it is out of its depth. It is **not** a better raw
//! similarity search than HNSW or learned embeddings.
//!
//! ```
//! use flyrag::{FlyRag, Outcome, RagParams};
//! # fn main() -> flyrag::Result<()> {
//! let mut rag = FlyRag::new(RagParams::prereg(7))?;
//! let bun = rag.add("install the dependencies with bun install, not npm install")?;
//! let npm = rag.add("install the dependencies with npm install and npm ci")?;
//! let a = rag.query("how do I install the dependencies", 5)?;
//! // Feedback names a document from this offer (credit is forward-only).
//! for hit in &a.hits {
//!     let outcome = if hit.doc == bun {
//!         Outcome::Helped
//!     } else {
//!         Outcome::Misled
//!     };
//!     rag.feedback(a.offer, hit.doc, outcome)?;
//! }
//! let b = rag.query("how do I install the dependencies", 5)?;
//! assert_eq!(b.hits[0].doc, bun);
//! # let _ = npm;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod rag;
mod readout;
mod text;

pub use error::{FlyRagError, Result};
pub use rag::{Answer, DocId, FlyRag, Learned, OfferId, Outcome, RagParams, Ranked};
pub use readout::ReadoutParams;
pub use text::TextEncoder;
