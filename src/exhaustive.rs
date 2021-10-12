//! [Exhaustive nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Linear_search).

use crate::distance::Proximity;
use crate::knn::{ExactNeighbors, NearestNeighbors, Neighborhood};

use std::iter::FromIterator;

/// A [`NearestNeighbors`] implementation that does exhaustive search.
#[derive(Clone, Debug)]
pub struct ExhaustiveSearch<T>(Vec<T>);

impl<T> ExhaustiveSearch<T> {
    /// Create an empty ExhaustiveSearch index.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Iterate over the items stored in this index.
    pub fn iter(&self) -> Iter<T> {
        self.into_iter()
    }

    /// Get the size of this index.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if this index is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Add a new item to the index.
    pub fn push(&mut self, item: T) {
        self.0.push(item);
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

/// An iterator that moves values out of an exhaustive index.
#[derive(Debug)]
pub struct IntoIter<T>(std::vec::IntoIter<T>);

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.0.next()
    }
}

impl<T> IntoIterator for ExhaustiveSearch<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.0.into_iter())
    }
}

/// An iterator over the values in an exhaustive index.
#[derive(Debug)]
pub struct Iter<'a, T>(std::slice::Iter<'a, T>);

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.0.next()
    }
}

impl<'a, T> IntoIterator for &'a ExhaustiveSearch<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter(self.0.iter())
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

    use crate::knn::tests::test_exact_neighbors;

    #[test]
    fn test_exhaustive_index() {
        test_exact_neighbors(ExhaustiveSearch::from_iter);
    }
}
