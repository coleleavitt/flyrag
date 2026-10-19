//! Errors.

/// Everything that can go wrong in FlyRAG.
#[derive(Debug, thiserror::Error)]
pub enum FlyRagError {
    /// An error from the underlying kenyon structures.
    #[error(transparent)]
    Kenyon(#[from] kenyon::KenyonError),
    /// A parameter is outside its accepted range.
    #[error("invalid parameter {name}: {reason}")]
    InvalidParameter {
        /// The parameter.
        name: &'static str,
        /// What was wrong.
        reason: String,
    },
    /// Feedback for an offer this store never made, or one already evicted.
    #[error("unknown offer {0}")]
    UnknownOffer(u64),
    /// Feedback for a document that was not in the offer (credit is
    /// forward-only: only documents shown before the outcome can learn).
    #[error("document {doc} was not offered in offer {offer}")]
    NotOffered {
        /// The offer.
        offer: u64,
        /// The document.
        doc: u32,
    },
    /// A second outcome for the same document in the same offer.
    #[error("document {doc} already has an outcome in offer {offer}")]
    AlreadyAnswered {
        /// The offer.
        offer: u64,
        /// The document.
        doc: u32,
    },
    /// The store holds as many documents as a `u32` id can name.
    #[error("the store is full")]
    Full,
}

/// Result alias.
pub type Result<T> = std::result::Result<T, FlyRagError>;

pub(crate) fn invalid(name: &'static str, reason: impl Into<String>) -> FlyRagError {
    FlyRagError::InvalidParameter {
        name,
        reason: reason.into(),
    }
}
