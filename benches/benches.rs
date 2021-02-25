//! Benchmark for various NearestNeighbors implementations.

use acap::euclid::Euclidean;
use acap::exhaustive::ExhaustiveSearch;
use acap::kd::{FlatKdTree, KdTree};
use acap::knn::NearestNeighbors;
use acap::vp::{FlatVpTree, VpTree};

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use std::iter::FromIterator;

type Point = Euclidean<[f32; 3]>;

/// Generates a spiral used as the benchmark data set.
fn spiral() -> Vec<Point> {
    let mut points = Vec::new();

    let size = 1000;
    let turns = 10.0;
    for i in 0..size {
        let y = 2.0 * (i as f32) / (size as f32) - 1.0;
        let m = (1.0 - y * y).sqrt();
        let theta = turns * y * std::f32::consts::PI;
        let (sin, cos) = theta.sin_cos();
        let x = m * cos * cos;
        let z = m * sin * cos;
        points.push(Euclidean([x, y, z]));
    }

    points
}

fn bench_creation(c: &mut Criterion) {
    let points = black_box(spiral());

    let mut group = c.benchmark_group("Creation");

    macro_rules! bench {
        ($type:ident) => {
            group.bench_function(stringify!($type), |b| b.iter_batched(
                || points.clone(),
                |points| $type::from_iter(points),
                BatchSize::SmallInput,
            ));
        };
    }

    bench!(ExhaustiveSearch);
    bench!(VpTree);
    bench!(FlatVpTree);
    bench!(KdTree);
    bench!(FlatKdTree);

    group.finish();
}

fn bench_nearest_neighbors(c: &mut Criterion) {
    let points = black_box(spiral());
    let target = black_box(Euclidean([0.0, 0.0, 0.0]));

    macro_rules! bench {
        ($type:ident) => {
            let mut group = c.benchmark_group(stringify!($type));
            let index = $type::from_iter(points.clone());

            group.bench_function("nearest", |b| b.iter(
                || index.nearest(&target)
            ));
            group.bench_function("nearest_within", |b| b.iter(
                || index.nearest_within(&target, 0.1)
            ));
            group.bench_function("k_nearest", |b| b.iter(
                || index.k_nearest(&target, 3)
            ));
            group.bench_function("k_nearest_within", |b| b.iter(
                || index.k_nearest_within(&target, 3, 0.1)
            ));

            group.bench_function("merge_k_nearest", |b| b.iter_batched(
                || Vec::with_capacity(3),
                |mut n| {
                    index.merge_k_nearest(&target, 3, &mut n);
                    n
                },
                BatchSize::SmallInput,
            ));
            group.bench_function("merge_k_nearest_within", |b| b.iter_batched(
                || Vec::with_capacity(3),
                |mut n| {
                    index.merge_k_nearest_within(&target, 3, 0.1, &mut n);
                    n
                },
                BatchSize::SmallInput,
            ));

            group.finish();
        };
    }

    bench!(ExhaustiveSearch);
    bench!(VpTree);
    bench!(FlatVpTree);
    bench!(KdTree);
    bench!(FlatKdTree);
}

criterion_group!(benches, bench_creation, bench_nearest_neighbors);
criterion_main!(benches);
