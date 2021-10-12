//! Internal utilities.

use std::cmp::Ordering;

/// A wrapper that converts a partial ordering into a total one by panicking.
#[derive(Clone, Copy, Debug, PartialOrd)]
pub struct Ordered<T>(T);

impl<T> Ordered<T> {
    /// Wrap a value.
    pub fn new(item: T) -> Self {
        Self(item)
    }
}

impl<T> From<T> for Ordered<T> {
    fn from(item: T) -> Self {
        Self::new(item)
    }
}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl<T: PartialOrd> Ord for Ordered<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect("Comparison between unordered items")
    }
}

impl<T: PartialOrd> PartialEq for Ordered<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<T: PartialOrd> Eq for Ordered<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordered() {
        let one = Ordered::new(1.0);
        let two = Ordered::new(2.0);

        assert_eq!(one.cmp(&one), Ordering::Equal);
        assert_eq!(one.cmp(&two), Ordering::Less);
        assert_eq!(two.cmp(&one), Ordering::Greater);
    }

    #[test]
    #[should_panic(expected = "Comparison between unordered items")]
    fn test_unordered() {
        let one = Ordered::new(1.0);
        let nan = Ordered::new(f64::NAN);

        assert!(!(one < nan));
        assert!(!(nan < one));
        let _ = one.cmp(&nan);
    }
}
