use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Bowers, Radix2Dit, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{Field, TwoAdicField, scale_slice_in_place_single_core};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::pretty_name;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

// ---------------------------------------------------------------------------
// coset_shift_cols: sequential (old) vs parallel+SIMD (new)
// ---------------------------------------------------------------------------

/// Old implementation: sequential rows_mut with lazy shift.powers() and scalar multiply.
fn coset_shift_cols_old<F: Field>(mat: &mut RowMajorMatrix<F>, shift: F) {
    mat.rows_mut()
        .zip(shift.powers())
        .for_each(|(row, weight)| {
            row.iter_mut().for_each(|coeff| {
                *coeff *= weight;
            });
        });
}

/// New implementation: pre-compute powers, parallel rows, SIMD-packed scaling.
fn coset_shift_cols_new<F: Field>(mat: &mut RowMajorMatrix<F>, shift: F) {
    let powers: Vec<F> = shift.powers().take(mat.height()).collect();
    mat.par_rows_mut()
        .zip(powers.into_par_iter())
        .for_each(|(row, weight)| {
            scale_slice_in_place_single_core(row, weight);
        });
}

fn bench_coset_shift(c: &mut Criterion) {
    type F = BabyBear;

    let log_sizes: &[usize] = &[12, 13, 14, 16, 18];
    let ncols = 256;

    let mut group = c.benchmark_group(format!("coset_shift_cols/BabyBear/ncols={ncols}"));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    let shift = F::GENERATOR;

    for &log_n in log_sizes {
        let n = 1usize << log_n;
        let mat = RowMajorMatrix::<F>::rand(&mut rng, n, ncols);

        group.bench_with_input(BenchmarkId::new("old", n), &mat, |b, mat| {
            b.iter(|| {
                let mut m = mat.clone();
                coset_shift_cols_old(&mut m, shift);
                m
            });
        });

        group.bench_with_input(BenchmarkId::new("new", n), &mat, |b, mat| {
            b.iter(|| {
                let mut m = mat.clone();
                coset_shift_cols_new(&mut m, shift);
                m
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// coset_lde end-to-end: exercises coset_shift_cols, idft, and per-row scaling
// ---------------------------------------------------------------------------

fn bench_coset_lde<F, Dft>(c: &mut Criterion, log_sizes: &[usize], ncols: usize)
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "coset_lde/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        ncols,
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);

    for &log_n in log_sizes {
        let n = 1usize << log_n;
        let mat = RowMajorMatrix::<F>::rand(&mut rng, n, ncols);
        let dft = Dft::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.coset_lde_batch(mat.clone(), 1, F::GENERATOR);
            });
        });
    }

    group.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_coset_shift(c);

    let log_sizes: &[usize] = &[12, 13, 14, 16, 18];
    let ncols = 256;

    // Radix2Dit uses the default coset_lde_batch -> idft_batch + coset_dft_batch
    // which exercises both coset_shift_cols and reverse_matrix_rows.
    bench_coset_lde::<BabyBear, Radix2Dit<_>>(c, log_sizes, ncols);

    // Radix2Bowers has a custom coset_lde_batch with per-row scaling.
    bench_coset_lde::<BabyBear, Radix2Bowers>(c, log_sizes, ncols);

    // Radix2DitParallel has a fully custom coset_lde_batch (not affected
    // by coset_shift_cols changes, but useful as a reference).
    bench_coset_lde::<BabyBear, Radix2DitParallel<_>>(c, log_sizes, ncols);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
