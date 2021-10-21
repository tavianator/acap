//! [k-d trees](https://en.wikipedia.org/wiki/K-d_tree).

use crate::coords::Coordinates;
use crate::distance::Proximity;
use crate::lp::Minkowski;
use crate::knn::{ExactNeighbors, NearestNeighbors, Neighborhood};
use crate::util::Ordered;

use num_traits::Signed;

/// A node in a k-d tree.
#[derive(Debug)]
struct KdNode<T> {
    /// The item stored in this node.
    item: T,
    /// The left subtree, if any.
    left: Option<Box<Self>>,
    /// The right subtree, if any.
    right: Option<Box<Self>>,
}

impl<T: Coordinates> KdNode<T> {
    /// Create a new KdNode.
    fn new(item: T) -> Self {
        Self {
            item,
            left: None,
            right: None,
        }
    }

    /// Create a balanced tree.
    fn balanced<I: IntoIterator<Item = T>>(items: I) -> Option<Self> {
        let mut nodes: Vec<_> = items
            .into_iter()
            .map(Self::new)
            .map(Box::new)
            .map(Some)
            .collect();

        Self::balanced_recursive(&mut nodes, 0)
            .map(|node| *node)
    }

    /// Create a balanced subtree.
    fn balanced_recursive(nodes: &mut [Option<Box<Self>>], level: usize) -> Option<Box<Self>> {
        if nodes.is_empty() {
            return None;
        }

        nodes.sort_unstable_by_key(|x| Ordered::new(x.as_ref().unwrap().item.coord(level)));

        let (left, right) = nodes.split_at_mut(nodes.len() / 2);
        let (node, right) = right.split_first_mut().unwrap();
        let mut node = node.take().unwrap();

        let next = (level + 1) % node.item.dims();
        node.left = Self::balanced_recursive(left, next);
        node.right = Self::balanced_recursive(right, next);

        Some(node)
    }

    /// Push a new item into this subtree.
    fn push(&mut self, item: T, level: usize) {
        let next = (level + 1) % item.dims();

        if item.coord(level) <= self.item.coord(level) {
            if let Some(left) = &mut self.left {
                left.push(item, next);
            } else {
                self.left = Some(Box::new(Self::new(item)));
            }
        } else {
            if let Some(right) = &mut self.right {
                right.push(item, next);
            } else {
                self.right = Some(Box::new(Self::new(item)));
            }
        }
    }
}

/// Marker trait for [`Proximity`] implementations that are compatible with k-d trees.
pub trait KdProximity<V: ?Sized = Self>
where
    Self: Coordinates<Value = V::Value>,
    Self: Proximity<V>,
    Self::Value: PartialOrd<Self::Distance>,
    V: Coordinates,
{}

/// Blanket [`KdProximity`] implementation.
impl<K, V> KdProximity<V> for K
where
    K: Coordinates<Value = V::Value>,
    K: Proximity<V>,
    K::Value: PartialOrd<K::Distance>,
    V: Coordinates,
{}

trait KdSearch<K, V, N>: Copy
where
    K: KdProximity<V>,
    K::Value: PartialOrd<K::Distance>,
    V: Coordinates + Copy,
    N: Neighborhood<K, V>,
{
    /// Get this node's item.
    fn item(self) -> V;

    /// Get the left subtree.
    fn left(self) -> Option<Self>;

    /// Get the right subtree.
    fn right(self) -> Option<Self>;

    /// Recursively search for nearest neighbors.
    fn search(self, level: usize, neighborhood: &mut N) {
        let item = self.item();
        neighborhood.consider(item);

        let target = neighborhood.target();

        let bound = target.coord(level) - item.coord(level);
        let (near, far) = if bound.is_negative() {
            (self.left(), self.right())
        } else {
            (self.right(), self.left())
        };

        let next = (level + 1) % self.item().dims();

        if let Some(near) = near {
            near.search(next, neighborhood);
        }

        if let Some(far) = far {
            if neighborhood.contains(bound.abs()) {
                far.search(next, neighborhood);
            }
        }
    }
}

impl<'a, K, V, N> KdSearch<K, &'a V, N> for &'a KdNode<V>
where
    K: KdProximity<&'a V>,
    K::Value: PartialOrd<K::Distance>,
    V: Coordinates,
    N: Neighborhood<K, &'a V>,
{
    fn item(self) -> &'a V {
        &self.item
    }

    fn left(self) -> Option<Self> {
        self.left.as_deref()
    }

    fn right(self) -> Option<Self> {
        self.right.as_deref()
    }
}

/// A [k-d tree](https://en.wikipedia.org/wiki/K-d_tree).
#[derive(Debug)]
pub struct KdTree<T> {
    root: Option<KdNode<T>>,
}

impl<T: Coordinates> KdTree<T> {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self { root: None }
    }

    /// Create a balanced tree out of a sequence of items.
    pub fn balanced<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self {
            root: KdNode::balanced(items),
        }
    }

    /// Iterate over the items stored in this tree.
    pub fn iter(&self) -> Iter<'_, T> {
        self.into_iter()
    }

    /// Rebalance this k-d tree.
    pub fn balance(&mut self) {
        let mut nodes = Vec::new();
        if let Some(root) = self.root.take() {
            nodes.push(Some(Box::new(root)));
        }

        let mut i = 0;
        while i < nodes.len() {
            let node = nodes[i].as_mut().unwrap();
            let inside = node.left.take();
            let outside = node.right.take();
            if inside.is_some() {
                nodes.push(inside);
            }
            if outside.is_some() {
                nodes.push(outside);
            }

            i += 1;
        }

        self.root = KdNode::balanced_recursive(&mut nodes, 0)
            .map(|node| *node);
    }

    /// Push a new item into the tree.
    ///
    /// Inserting elements individually tends to unbalance the tree.  Use [`KdTree::balanced()`] if
    /// possible to create a balanced tree from a batch of items.
    pub fn push(&mut self, item: T) {
        if let Some(root) = &mut self.root {
            root.push(item, 0);
        } else {
            self.root = Some(KdNode::new(item));
        }
    }
}

impl<T: Coordinates> Default for KdTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Coordinates> Extend<T> for KdTree<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        if self.root.is_some() {
            for item in items {
                self.push(item);
            }
        } else {
            self.root = KdNode::balanced(items);
        }
    }
}

impl<T: Coordinates> FromIterator<T> for KdTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self::balanced(items)
    }
}

/// An iterator that moves values out of a k-d tree.
#[derive(Debug)]
pub struct IntoIter<T> {
    stack: Vec<KdNode<T>>,
}

impl<T> IntoIter<T> {
    fn new(node: Option<KdNode<T>>) -> Self {
        Self {
            stack: node.into_iter().collect(),
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|node| {
            if let Some(left) = node.left {
                self.stack.push(*left);
            }
            if let Some(right) = node.right {
                self.stack.push(*right);
            }
            node.item
        })
    }
}

impl<T> IntoIterator for KdTree<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.root)
    }
}

/// An iterator over the values in a k-d tree.
#[derive(Debug)]
pub struct Iter<'a, T> {
    stack: Vec<&'a KdNode<T>>,
}

impl<'a, T> Iter<'a, T> {
    fn new(node: &'a Option<KdNode<T>>) -> Self {
        Self {
            stack: node.as_ref().into_iter().collect(),
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|node| {
            if let Some(left) = &node.left {
                self.stack.push(left);
            }
            if let Some(right) = &node.right {
                self.stack.push(right);
            }
            &node.item
        })
    }
}

impl<'a, T> IntoIterator for &'a KdTree<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(&self.root)
    }
}

impl<K, V> NearestNeighbors<K, V> for KdTree<V>
where
    K: KdProximity<V>,
    K::Value: PartialOrd<K::Distance>,
    V: Coordinates,
{
    fn search<'k, 'v, N>(&'v self, mut neighborhood: N) -> N
    where
        K: 'k,
        V: 'v,
        N: Neighborhood<&'k K, &'v V>,
    {
        if let Some(root) = &self.root {
            root.search(0, &mut neighborhood);
        }
        neighborhood
    }
}

/// k-d trees are exact for [Minkowski] distances.
impl<K, V> ExactNeighbors<K, V> for KdTree<V>
where
    K: KdProximity<V> + Minkowski<V>,
    K::Value: PartialOrd<K::Distance>,
    V: Coordinates,
{}

/// A node in a flat k-d tree.
#[derive(Debug)]
struct FlatKdNode<T> {
    /// The item stored in this node.
    item: T,
    /// The size of the left subtree.
    left_len: usize,
}

impl<T: Coordinates> FlatKdNode<T> {
    /// Create a new FlatKdNode.
    fn new(item: T) -> Self {
        Self {
            item,
            left_len: 0,
        }
    }

    /// Create a balanced tree.
    fn balanced<I: IntoIterator<Item = T>>(items: I) -> Vec<Self> {
        let mut nodes: Vec<_> = items
            .into_iter()
            .map(Self::new)
            .collect();

        Self::balance_recursive(&mut nodes, 0);

        nodes
    }

    /// Create a balanced subtree.
    fn balance_recursive(nodes: &mut [Self], level: usize) {
        if !nodes.is_empty() {
            nodes.sort_unstable_by_key(|x| Ordered::new(x.item.coord(level)));

            let mid = nodes.len() / 2;
            nodes.swap(0, mid);

            let (node, children) = nodes.split_first_mut().unwrap();
            let (left, right) = children.split_at_mut(mid);
            node.left_len = left.len();

            let next = (level + 1) % node.item.dims();
            Self::balance_recursive(left, next);
            Self::balance_recursive(right, next);
        }
    }
}

impl<'a, K, V, N> KdSearch<K, &'a V, N> for &'a [FlatKdNode<V>]
where
    K: KdProximity<&'a V>,
    K::Value: PartialOrd<K::Distance>,
    V: Coordinates,
    N: Neighborhood<K, &'a V>,
{
    fn item(self) -> &'a V {
        &self[0].item
    }

    fn left(self) -> Option<Self> {
        let end = self[0].left_len + 1;
        if end > 1 {
            Some(&self[1..end])
        } else {
            None
        }
    }

    fn right(self) -> Option<Self> {
        let start = self[0].left_len + 1;
        if start < self.len() {
            Some(&self[start..])
        } else {
            None
        }
    }
}

/// A [k-d tree] stored as a flat array.
///
/// A FlatKdTree is always balanced and usually more efficient than a [`KdTree`], but doesn't
/// support dynamic updates.
///
/// [k-d tree]: https://en.wikipedia.org/wiki/K-d_tree
#[derive(Debug)]
pub struct FlatKdTree<T> {
    nodes: Vec<FlatKdNode<T>>,
}

impl<T: Coordinates> FlatKdTree<T> {
    /// Create a balanced tree out of a sequence of items.
    pub fn balanced<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self {
            nodes: FlatKdNode::balanced(items),
        }
    }

    /// Iterate over the items stored in this tree.
    pub fn iter(&self) -> FlatIter<'_, T> {
        self.into_iter()
    }
}

impl<T: Coordinates> FromIterator<T> for FlatKdTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self::balanced(items)
    }
}

/// An iterator that moves values out of a flat k-d tree.
#[derive(Debug)]
pub struct FlatIntoIter<T>(std::vec::IntoIter<FlatKdNode<T>>);

impl<T> Iterator for FlatIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|n| n.item)
    }
}

impl<T> IntoIterator for FlatKdTree<T> {
    type Item = T;
    type IntoIter = FlatIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        FlatIntoIter(self.nodes.into_iter())
    }
}

/// An iterator over the values in a flat k-d tree.
#[derive(Debug)]
pub struct FlatIter<'a, T>(std::slice::Iter<'a, FlatKdNode<T>>);

impl<'a, T> Iterator for FlatIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|n| &n.item)
    }
}

impl<'a, T> IntoIterator for &'a FlatKdTree<T> {
    type Item = &'a T;
    type IntoIter = FlatIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        FlatIter(self.nodes.iter())
    }
}

impl<K, V> NearestNeighbors<K, V> for FlatKdTree<V>
where
    K: KdProximity<V>,
    K::Value: PartialOrd<K::Distance>,
    V: Coordinates,
{
    fn search<'k, 'v, N>(&'v self, mut neighborhood: N) -> N
    where
        K: 'k,
        V: 'v,
        N: Neighborhood<&'k K, &'v V>,
    {
        if !self.nodes.is_empty() {
            self.nodes.as_slice().search(0, &mut neighborhood);
        }
        neighborhood
    }
}

/// k-d trees are exact for [Minkowski] distances.
impl<K, V> ExactNeighbors<K, V> for FlatKdTree<V>
where
    K: KdProximity<V> + Minkowski<V>,
    K::Value: PartialOrd<K::Distance>,
    V: Coordinates,
{}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::knn::tests::test_exact_neighbors;

    #[test]
    fn test_kd_tree() {
        test_exact_neighbors(KdTree::from_iter);
    }

    #[test]
    fn test_unbalanced_kd_tree() {
        test_exact_neighbors(|points| {
            let mut tree = KdTree::new();
            for point in points {
                tree.push(point);
            }
            tree
        });
    }

    #[test]
    fn test_flat_kd_tree() {
        test_exact_neighbors(FlatKdTree::from_iter);
    }
}
