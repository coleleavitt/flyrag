//! The learned part: one document's two readout banks over KC units
//! (research/flyrag/PREREG.md "Readout", "Update", "Decay", amendment A1).
//!
//! Each bank weight starts at the prior `P`. A verified outcome depresses one
//! bank on the query's active units (helped → "skip", misled → "use"); nothing
//! ever raises a bank except decay back toward `P`. So unlearning is an
//! opposing trace, never an overwrite, and `learned` is bounded in [−1, 1].

use std::collections::HashMap;

/// Readout constants (the prereg's values are [`ReadoutParams::PREREG`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadoutParams {
    /// Prior bank weight `P`.
    pub prior: f32,
    /// Floor `F < P`.
    pub floor: f32,
    /// Learning rate `η`.
    pub eta: f32,
    /// Largest step per outcome.
    pub cap: f32,
    /// `|δ|` below this teaches nothing (an expected outcome).
    pub min_delta: f32,
    /// Half-life of decay toward the prior, in queries served (0 = none).
    pub half_life: u64,
}

impl ReadoutParams {
    /// The preregistered constants.
    pub const PREREG: Self = Self {
        prior: 0.5,
        floor: 0.05,
        eta: 0.3,
        cap: 0.25,
        min_delta: 0.05,
        half_life: 2_000,
    };
}

/// One bank weight with the clock it was last set at (decay is lazy).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cell {
    w: f32,
    at: u64,
}

/// Which bank an outcome depresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bank {
    Use,
    Skip,
}

/// A document's two banks; untouched units are implicitly at the prior.
#[derive(Debug, Clone, Default)]
pub(crate) struct Banks {
    use_w: HashMap<u32, Cell>,
    skip_w: HashMap<u32, Cell>,
}

impl Banks {
    fn value(p: &ReadoutParams, cell: Option<&Cell>, now: u64) -> f32 {
        let Some(c) = cell else { return p.prior };
        if p.half_life == 0 || now <= c.at {
            return c.w;
        }
        let age = (now - c.at) as f64 / p.half_life as f64;
        let gap = f64::from(p.prior - c.w) * 0.5f64.powf(age);
        (f64::from(p.prior) - gap) as f32
    }

    /// `mean over units of (w_use − w_skip) / (P − F)`, in [−1, 1].
    pub(crate) fn learned(&self, p: &ReadoutParams, units: &[u32], now: u64) -> f32 {
        if units.is_empty() || self.use_w.is_empty() && self.skip_w.is_empty() {
            return 0.0;
        }
        let s: f32 = units
            .iter()
            .map(|u| {
                Self::value(p, self.use_w.get(u), now) - Self::value(p, self.skip_w.get(u), now)
            })
            .sum();
        (s / units.len() as f32 / (p.prior - p.floor)).clamp(-1.0, 1.0)
    }

    /// Depress `bank` on `units` by step `s`: `w ← max(F, w (1 − s))`.
    pub(crate) fn depress(
        &mut self,
        p: &ReadoutParams,
        bank: Bank,
        units: &[u32],
        s: f32,
        now: u64,
    ) {
        let map = match bank {
            Bank::Use => &mut self.use_w,
            Bank::Skip => &mut self.skip_w,
        };
        for &u in units {
            let cur = Self::value(p, map.get(&u), now);
            map.insert(
                u,
                Cell {
                    w: (cur * (1.0 - s)).max(p.floor),
                    at: now,
                },
            );
        }
    }

    /// Units with a bank weight away from the prior (diagnostics).
    pub(crate) fn moved_units(&self, p: &ReadoutParams, now: u64) -> usize {
        let off = |m: &HashMap<u32, Cell>| {
            m.values()
                .filter(|c| (Self::value(p, Some(c), now) - p.prior).abs() > 1e-6)
                .count()
        };
        off(&self.use_w) + off(&self.skip_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: ReadoutParams = ReadoutParams::PREREG;

    #[test]
    fn untouched_banks_are_neutral_and_updates_are_bounded() {
        let mut b = Banks::default();
        let units = [1, 2, 3];
        assert_eq!(b.learned(&P, &units, 0), 0.0);
        for _ in 0..1_000 {
            b.depress(&P, Bank::Skip, &units, P.cap, 0);
        }
        // Skip at the floor, use at the prior: learned is exactly +1.
        assert!((b.learned(&P, &units, 0) - 1.0).abs() < 1e-6);
        b.depress(&P, Bank::Use, &units, P.cap, 0);
        assert!(
            b.learned(&P, &units, 0) < 1.0,
            "the opposing trace lowers it"
        );
        assert!(b.learned(&P, &[9], 0).abs() < 1e-6, "other units untouched");
    }

    #[test]
    fn decay_returns_to_the_prior_and_never_overshoots() {
        let mut b = Banks::default();
        b.depress(&P, Bank::Skip, &[1], 0.25, 0);
        let fresh = b.learned(&P, &[1], 0);
        let half = b.learned(&P, &[1], P.half_life);
        assert!((half - fresh / 2.0).abs() < 1e-4, "{half} vs {fresh}/2");
        let late = b.learned(&P, &[1], 100 * P.half_life);
        assert!((0.0..1e-6).contains(&late));
        assert_eq!(b.moved_units(&P, 100 * P.half_life), 0);
    }
}
