use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_monty_31::dft::RecursiveDft;

/// An enum containing several different options for discrete Fourier Transform.
///
/// This implements `TwoAdicSubgroupDft` by passing to whatever the contained struct is.
#[derive(Clone, Debug)]
pub enum DftChoice<F> {
    Recursive(RecursiveDft<F>),
    Parallel(Radix2DitParallel<F>),
    SmallBatch(Radix2DFTSmallBatch<F>),
}

impl<F: Default> Default for DftChoice<F> {
    // We have to fix a default for the `TwoAdicSubgroupDft` trait. We choose `Radix2DitParallel` as one of the features
    // of `RecursiveDft` is that it works better when initialized with knowledge of the expected size.
    fn default() -> Self {
        Self::Parallel(Radix2DitParallel::<F>::default())
    }
}

#[allow(refining_impl_trait_reachable)]
impl<F: TwoAdicField + Ord> TwoAdicSubgroupDft<F> for DftChoice<F>
where
    RecursiveDft<F>: TwoAdicSubgroupDft<F>,
{
    #[inline]
    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        match self {
            Self::Recursive(inner_dft) => inner_dft.dft_batch(mat).to_row_major_matrix(),
            Self::Parallel(inner_dft) => inner_dft.dft_batch(mat).to_row_major_matrix(),
            Self::SmallBatch(inner_dft) => inner_dft.dft_batch(mat).to_row_major_matrix(),
        }
    }

    #[inline]
    fn coset_dft_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        match self {
            Self::Recursive(inner_dft) => inner_dft.coset_dft_batch(mat, shift).to_row_major_matrix(),
            Self::Parallel(inner_dft) => inner_dft.coset_dft_batch(mat, shift).to_row_major_matrix(),
            Self::SmallBatch(inner_dft) => inner_dft.coset_dft_batch(mat, shift).to_row_major_matrix(),
        }
    }

    #[inline]
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<F> {
        match self {
            Self::Recursive(inner_dft) => inner_dft
                .coset_lde_batch(mat, added_bits, shift)
                .to_row_major_matrix(),
            Self::Parallel(inner_dft) => inner_dft
                .coset_lde_batch(mat, added_bits, shift)
                .to_row_major_matrix(),
            Self::SmallBatch(inner_dft) => inner_dft
                .coset_lde_batch(mat, added_bits, shift)
                .to_row_major_matrix(),
        }
    }
}
