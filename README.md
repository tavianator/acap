`acap`
======

[![crates.io](https://img.shields.io/crates/v/acap.svg)](https://crates.io/crates/acap)
[![Documentation](https://docs.rs/acap/badge.svg)](https://docs.rs/acap)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tavianator/knn/blob/master/LICENSE)
[![Build Status](https://travis-ci.com/tavianator/acap.svg?branch=master)](https://travis-ci.com/tavianator/acap)

As Close As Possible — [nearest neighbor search] in Rust.

[nearest neighbor search]: https://en.wikipedia.org/wiki/Nearest_neighbor_search


Example
-------

```rust
use acap::euclid::Euclidean;
use acap::vp::VpTree;
use acap::NearestNeighbors;

let tree = VpTree::balanced(vec![
    Euclidean([3, 4]),
    Euclidean([5, 12]),
    Euclidean([8, 15]),
    Euclidean([7, 24]),
]);

let nearest = tree.nearest(&[7, 7]).unwrap();
assert_eq!(nearest.item, &Euclidean([3, 4]));
assert_eq!(nearest.distance, 5);
```
