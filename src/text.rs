//! Text → feature vector (research/flyrag/PREREG.md, "Text features").
//!
//! Lower-cased alphanumeric tokens; unigrams and adjacent bigrams hashed with a
//! seeded FNV-1a into `dim` non-negative buckets; term weight `1 + ln(tf)`.
//! Non-negative weights keep the input a "firing rate", as the projection
//! neurons' are.

/// A seeded, hashing text encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextEncoder {
    dim: usize,
    seed: u64,
}

impl TextEncoder {
    /// An encoder into `dim` buckets (`dim >= 1`).
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            dim: dim.max(1),
            seed,
        }
    }

    /// Output dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The lower-cased alphanumeric tokens of `text`.
    pub fn tokens(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect()
    }

    fn bucket(&self, parts: &[&str]) -> usize {
        // FNV-1a over the seed, then each part with a separator byte.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ self.seed;
        for (i, p) in parts.iter().enumerate() {
            if i > 0 {
                h ^= 0x1f;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            for b in p.bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
        (h % self.dim as u64) as usize
    }

    /// The distinct buckets `text`'s features fall in, sorted.
    pub fn buckets(&self, text: &str) -> Vec<usize> {
        let toks = Self::tokens(text);
        let mut b: Vec<usize> = toks.iter().map(|t| self.bucket(&[t])).collect();
        b.extend(toks.windows(2).map(|w| self.bucket(&[&w[0], &w[1]])));
        b.sort_unstable();
        b.dedup();
        b
    }

    /// [`TextEncoder::encode`] with each bucket multiplied by `weights[b]`
    /// (missing weights count as 1).
    pub fn encode_weighted(&self, text: &str, weights: &[f32]) -> Vec<f32> {
        let mut x = self.encode(text);
        for (v, w) in x.iter_mut().zip(weights) {
            *v *= w;
        }
        x
    }

    /// The feature vector of `text`.
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let toks = Self::tokens(text);
        let mut tf = vec![0u32; self.dim];
        for t in &toks {
            tf[self.bucket(&[t])] += 1;
        }
        for w in toks.windows(2) {
            tf[self.bucket(&[&w[0], &w[1]])] += 1;
        }
        tf.into_iter()
            .map(|c| if c == 0 { 0.0 } else { 1.0 + (c as f32).ln() })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_and_weights() {
        assert_eq!(
            TextEncoder::tokens("Use bun, not NPM!"),
            ["use", "bun", "not", "npm"]
        );
        let e = TextEncoder::new(64, 1);
        let x = e.encode("bun bun npm");
        assert_eq!(x.len(), 64);
        assert!(x.iter().all(|&v| v >= 0.0));
        let max = x.iter().copied().fold(0.0f32, f32::max);
        assert!((max - (1.0 + 2f32.ln())).abs() < 1e-6, "tf 2 → 1 + ln 2");
        assert_eq!(e.encode("bun bun npm"), x, "deterministic");
        assert_ne!(TextEncoder::new(64, 2).encode("bun bun npm"), x, "seeded");
        assert!(e.encode("").iter().all(|&v| v == 0.0));
    }
}
