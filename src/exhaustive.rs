//! Exhaustive nearest neighbor search.

use crate::distance::Proximity;
use crate::{ExactNeighbors, NearestNeighbors, Neighborhood};

use std::iter::FromIterator;

/// A [`NearestNeighbors`] implementation that does exhaustive search.
#[derive(Debug)]
pub struct ExhaustiveSearch<T>(Vec<T>);

impl<T> ExhaustiveSearch<T> {
    /// Create an empty ExhaustiveSearch index.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Add a new item to the index.
    pub fn push(&mut self, item: T) {
        self.0.push(item);
    }

    /// Get the size of this index.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if this index is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T> Default for ExhaustiveSearch<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FromIterator<T> for ExhaustiveSearch<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self(items.into_iter().collect())
    }
}

impl<T> IntoIterator for ExhaustiveSearch<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> Extend<T> for ExhaustiveSearch<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push(value);
        }
    }
}

impl<K: Proximity<V>, V> NearestNeighbors<K, V> for ExhaustiveSearch<V> {
    fn search<'k, 'v, N>(&'v self, mut neighborhood: N) -> N
    where
        K: 'k,
        V: 'v,
        N: Neighborhood<&'k K, &'v V>,
    {
        for e in &self.0 {
            neighborhood.consider(e);
        }
        neighborhood
    }
}

impl<K: Proximity<V>, V> ExactNeighbors<K, V> for ExhaustiveSearch<V> {}

#[cfg(test)]
pub mod tests {
    use super::*;

    use crate::tests::test_nearest_neighbors;

    #[test]
    fn test_exhaustive_index() {
        test_nearest_neighbors(ExhaustiveSearch::from_iter);
    }
}
