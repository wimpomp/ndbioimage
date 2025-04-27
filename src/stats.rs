use anyhow::{anyhow, Result};
use ndarray::{Array, ArrayD, ArrayView, Axis, Dimension, RemoveAxis};

/// a trait to define the min, max, sum and mean operations along an axis
pub trait MinMax {
    type Output;

    fn max(self, axis: usize) -> Result<Self::Output>;
    fn min(self, axis: usize) -> Result<Self::Output>;
    fn sum(self, axis: usize) -> Result<Self::Output>;
    fn mean(self, axis: usize) -> Result<Self::Output>;
}

macro_rules! impl_frame_stats_float_view {
    ($($t:tt),+ $(,)?) => {
        $(
            impl<D> MinMax for ArrayView<'_, $t, D>
            where
                D: Dimension + RemoveAxis,
            {
                type Output = Array<$t, D::Smaller>;

                fn max(self, axis: usize) -> Result<Self::Output> {
                    let a: Vec<_> = self
                        .lanes(Axis(axis))
                        .into_iter()
                        .map(|x| {
                            x.iter()
                                .fold($t::NEG_INFINITY, |prev, curr| prev.max(*curr))
                        })
                        .collect();
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    Ok(ArrayD::from_shape_vec(shape, a)?.into_dimensionality()?)
                }

                fn min(self, axis: usize) -> Result<Self::Output> {
                    let a: Vec<_> = self
                        .lanes(Axis(axis))
                        .into_iter()
                        .map(|x| {
                            x.iter()
                                .fold($t::INFINITY, |prev, curr| prev.min(*curr))
                        })
                        .collect();
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    Ok(ArrayD::from_shape_vec(shape, a)?.into_dimensionality()?)
                }

                fn sum(self, axis: usize) -> Result<Self::Output> {
                    Ok(self.sum_axis(Axis(axis)))
                }

                fn mean(self, axis: usize) -> Result<Self::Output> {
                    self.mean_axis(Axis(axis)).ok_or_else(|| anyhow!("no mean"))
                }
            }
        )*
    };
}

macro_rules! impl_frame_stats_int_view {
    ($($t:tt),+ $(,)?) => {
        $(
            impl<D> MinMax for ArrayView<'_, $t, D>
            where
                D: Dimension + RemoveAxis,
            {
                type Output = Array<$t, D::Smaller>;

                fn max(self, axis: usize) -> Result<Self::Output> {
                    let a: Vec<_> = self
                        .lanes(Axis(axis))
                        .into_iter()
                        .map(|x| *x.iter().max().unwrap())
                        .collect();
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    Ok(ArrayD::from_shape_vec(shape, a)?.into_dimensionality()?)
                }

                fn min(self, axis: usize) -> Result<Self::Output> {
                    let a: Vec<_> = self
                        .lanes(Axis(axis))
                        .into_iter()
                        .map(|x| *x.iter().min().unwrap())
                        .collect();
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    Ok(ArrayD::from_shape_vec(shape, a)?.into_dimensionality()?)
                }

                fn sum(self, axis: usize) -> Result<Self::Output> {
                    Ok(self.sum_axis(Axis(axis)))
                }

                fn mean(self, axis: usize) -> Result<Self::Output> {
                    self.mean_axis(Axis(axis)).ok_or_else(|| anyhow!("no mean"))
                }
            }
        )*
    };
}

macro_rules! impl_frame_stats_float {
    ($($t:tt),+ $(,)?) => {
        $(
            impl<D> MinMax for Array<$t, D>
            where
                D: Dimension + RemoveAxis,
            {
                type Output = Array<$t, D::Smaller>;

                fn max(self, axis: usize) -> Result<Self::Output> {
                    let a: Vec<_> = self
                        .lanes(Axis(axis))
                        .into_iter()
                        .map(|x| {
                            x.iter()
                                .fold($t::NEG_INFINITY, |prev, curr| prev.max(*curr))
                        })
                        .collect();
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    Ok(ArrayD::from_shape_vec(shape, a)?.into_dimensionality()?)
                }

                fn min(self, axis: usize) -> Result<Self::Output> {
                    let a: Vec<_> = self
                        .lanes(Axis(axis))
                        .into_iter()
                        .map(|x| {
                            x.iter()
                                .fold($t::INFINITY, |prev, curr| prev.min(*curr))
                        })
                        .collect();
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    Ok(ArrayD::from_shape_vec(shape, a)?.into_dimensionality()?)
                }

                fn sum(self, axis: usize) -> Result<Self::Output> {
                    Ok(self.sum_axis(Axis(axis)))
                }

                fn mean(self, axis: usize) -> Result<Self::Output> {
                    self.mean_axis(Axis(axis)).ok_or_else(|| anyhow!("no mean"))
                }
            }
        )*
    };
}

macro_rules! impl_frame_stats_int {
    ($($t:tt),+ $(,)?) => {
        $(
            impl<D> MinMax for Array<$t, D>
            where
                D: Dimension + RemoveAxis,
            {
                type Output = Array<$t, D::Smaller>;

                fn max(self, axis: usize) -> Result<Self::Output> {
                    let a: Vec<_> = self
                        .lanes(Axis(axis))
                        .into_iter()
                        .map(|x| *x.iter().max().unwrap())
                        .collect();
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    Ok(ArrayD::from_shape_vec(shape, a)?.into_dimensionality()?)
                }

                fn min(self, axis: usize) -> Result<Self::Output> {
                    let a: Vec<_> = self
                        .lanes(Axis(axis))
                        .into_iter()
                        .map(|x| *x.iter().min().unwrap())
                        .collect();
                    let mut shape = self.shape().to_vec();
                    shape.remove(axis);
                    Ok(ArrayD::from_shape_vec(shape, a)?.into_dimensionality()?)
                }

                fn sum(self, axis: usize) -> Result<Self::Output> {
                    Ok(self.sum_axis(Axis(axis)))
                }

                fn mean(self, axis: usize) -> Result<Self::Output> {
                    self.mean_axis(Axis(axis)).ok_or_else(|| anyhow!("no mean"))
                }
            }
        )*
    };
}

impl_frame_stats_float_view!(f32, f64);
impl_frame_stats_int_view!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize);
impl_frame_stats_float!(f32, f64);
impl_frame_stats_int!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize);
