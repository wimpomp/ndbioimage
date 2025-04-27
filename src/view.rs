use crate::axes::{slice_info, Ax, Axis, Operation, Slice, SliceInfoElemDef};
use crate::reader::Reader;
use crate::stats::MinMax;
use anyhow::{anyhow, Error, Result};
use indexmap::IndexMap;
use itertools::iproduct;
use ndarray::{
    s, Array, Array0, Array1, Array2, ArrayD, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix5, IxDyn,
    SliceArg, SliceInfoElem,
};
use num::{Bounded, FromPrimitive, Zero};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::any::type_name;
use std::collections::HashMap;
use std::iter::Sum;
use std::marker::PhantomData;
use std::mem::{transmute, transmute_copy};
use std::ops::{AddAssign, Deref, Div};
use std::path::PathBuf;
use std::sync::Arc;

fn idx_bnd(idx: isize, bnd: isize) -> Result<isize> {
    if idx < -bnd {
        Err(anyhow!("Index {} out of bounds {}", idx, bnd))
    } else if idx < 0 {
        Ok(bnd - idx)
    } else if idx < bnd {
        Ok(idx)
    } else {
        Err(anyhow!("Index {} out of bounds {}", idx, bnd))
    }
}

fn slc_bnd(idx: isize, bnd: isize) -> Result<isize> {
    if idx < -bnd {
        Err(anyhow!("Index {} out of bounds {}", idx, bnd))
    } else if idx < 0 {
        Ok(bnd - idx)
    } else if idx <= bnd {
        Ok(idx)
    } else {
        Err(anyhow!("Index {} out of bounds {}", idx, bnd))
    }
}

pub trait Number:
    'static + AddAssign + Bounded + Clone + Div<Self, Output = Self> + FromPrimitive + PartialOrd + Zero
{
}
impl<T> Number for T where
    T: 'static
        + AddAssign
        + Bounded
        + Clone
        + Div<Self, Output = Self>
        + FromPrimitive
        + PartialOrd
        + Zero
{
}

/// sliceable view on an image file
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct View<D: Dimension> {
    reader: Arc<Reader>,
    #[serde_as(as = "Vec<SliceInfoElemDef>")]
    slice: Vec<SliceInfoElem>,
    axes: Vec<Axis>,
    operations: IndexMap<Axis, Operation>,
    dimensionality: PhantomData<D>,
}

impl<D: Dimension> View<D> {
    pub(crate) fn new(reader: Arc<Reader>, slice: Vec<SliceInfoElem>, axes: Vec<Axis>) -> Self {
        Self {
            reader,
            slice,
            axes,
            operations: IndexMap::new(),
            dimensionality: PhantomData,
        }
    }

    /// the file path
    pub fn path(&self) -> &PathBuf {
        &self.reader.path
    }

    /// the series in the file
    pub fn series(&self) -> i32 {
        self.reader.series
    }

    fn with_operations(mut self, operations: IndexMap<Axis, Operation>) -> Self {
        self.operations = operations;
        self
    }

    /// change the dimension into a dynamic dimension
    pub fn into_dyn(self) -> View<IxDyn> {
        View {
            reader: self.reader,
            slice: self.slice,
            axes: self.axes,
            operations: self.operations,
            dimensionality: PhantomData,
        }
    }

    /// change the dimension into a concrete dimension
    pub fn into_dimensionality<D2: Dimension>(self) -> Result<View<D2>> {
        if let Some(d) = D2::NDIM {
            if d == self.ndim() {
                Ok(View {
                    reader: self.reader,
                    slice: self.slice,
                    axes: self.axes,
                    operations: self.operations,
                    dimensionality: PhantomData,
                })
            } else {
                Err(anyhow!("Dimensionality mismatch: {} != {}", d, self.ndim()))
            }
        } else {
            Ok(View {
                reader: self.reader,
                slice: self.slice,
                axes: self.axes,
                operations: self.operations,
                dimensionality: PhantomData,
            })
        }
    }

    /// the order of the axes, including axes sliced out
    pub fn get_axes(&self) -> &[Axis] {
        &self.axes
    }

    /// the slice defining the view
    pub fn get_slice(&self) -> &[SliceInfoElem] {
        &self.slice
    }

    /// the axes in the view
    pub fn axes(&self) -> Vec<Axis> {
        self.axes
            .iter()
            .zip(self.slice.iter())
            .filter_map(|(ax, s)| {
                if s.is_index() || self.operations.contains_key(ax) {
                    None
                } else {
                    Some(*ax)
                }
            })
            .collect()
    }

    pub(crate) fn op_axes(&self) -> Vec<Axis> {
        self.operations.keys().cloned().collect()
    }

    /// the number of dimensions in the view
    pub fn ndim(&self) -> usize {
        if let Some(d) = D::NDIM {
            d
        } else {
            self.shape().len()
        }
    }

    /// the number of pixels in the view
    pub fn size(&self) -> usize {
        self.shape().into_iter().product()
    }

    /// the shape of the view
    pub fn shape(&self) -> Vec<usize> {
        let mut shape = Vec::<usize>::new();
        for (ax, s) in self.axes.iter().zip(self.slice.iter()) {
            match s {
                SliceInfoElem::Slice { start, end, step } => {
                    if !self.operations.contains_key(ax) {
                        if let Some(e) = end {
                            shape.push(((e - start).max(0) / step) as usize);
                        } else {
                            panic!("slice has no end")
                        }
                    }
                }
                SliceInfoElem::Index(_) => {}
                SliceInfoElem::NewAxis => {
                    if !self.operations.contains_key(ax) {
                        shape.push(1);
                    }
                }
            }
        }
        shape
    }

    fn shape_all(&self) -> Vec<usize> {
        let mut shape = Vec::<usize>::new();
        for s in self.slice.iter() {
            match s {
                SliceInfoElem::Slice { start, end, step } => {
                    if let Some(e) = end {
                        shape.push(((e - start).max(0) / step) as usize);
                    } else {
                        panic!("slice has no end")
                    }
                }
                _ => shape.push(1),
            }
        }
        shape
    }

    /// swap two axes
    pub fn swap_axes<A: Ax>(&self, axis0: A, axis1: A) -> Result<Self> {
        let idx0 = axis0.pos_op(&self.axes, &self.slice, &self.op_axes())?;
        let idx1 = axis1.pos_op(&self.axes, &self.slice, &self.op_axes())?;
        let mut slice = self.slice.to_vec();
        slice.swap(idx0, idx1);
        let mut axes = self.axes.clone();
        axes.swap(idx0, idx1);
        Ok(View::new(self.reader.clone(), slice, axes))
    }

    /// subset of gives axes will be reordered in given order
    pub fn permute_axes<A: Ax>(&self, axes: &[A]) -> Result<Self> {
        let idx: Vec<usize> = axes
            .iter()
            .map(|a| a.pos_op(&self.axes, &self.slice, &self.op_axes()).unwrap())
            .collect();
        let mut jdx = idx.clone();
        jdx.sort();
        let mut slice = self.slice.to_vec();
        let mut axes = self.axes.clone();
        for (&i, j) in idx.iter().zip(jdx) {
            slice[j] = self.slice[i];
            axes[j] = self.axes[i];
        }
        Ok(View::new(self.reader.clone(), slice, axes))
    }

    /// reverse the order of the axes
    pub fn transpose(&self) -> Result<Self> {
        Ok(View::new(
            self.reader.clone(),
            self.slice.iter().rev().cloned().collect(),
            self.axes.iter().rev().cloned().collect(),
        ))
    }

    fn operate<A: Ax>(&self, axis: A, operation: Operation) -> Result<View<D::Smaller>> {
        let pos = axis.pos_op(&self.axes, &self.slice, &self.op_axes())?;
        let mut operations = self.operations.clone();
        operations.insert(self.axes[pos], operation);
        Ok(
            View::new(self.reader.clone(), self.slice.clone(), self.axes.clone())
                .with_operations(operations),
        )
    }

    /// maximum along axis
    pub fn max_proj<A: Ax>(&self, axis: A) -> Result<View<D::Smaller>> {
        self.operate(axis, Operation::Max)
    }

    /// minimum along axis
    pub fn min_proj<A: Ax>(&self, axis: A) -> Result<View<D::Smaller>> {
        self.operate(axis, Operation::Min)
    }

    /// sum along axis
    pub fn sum_proj<A: Ax>(&self, axis: A) -> Result<View<D::Smaller>> {
        self.operate(axis, Operation::Sum)
    }

    /// mean along axis
    pub fn mean_proj<A: Ax>(&self, axis: A) -> Result<View<D::Smaller>> {
        self.operate(axis, Operation::Mean)
    }

    /// created a new sliced view
    pub fn slice<I>(&self, info: I) -> Result<View<I::OutDim>>
    where
        I: SliceArg<D>,
    {
        if self.slice.out_ndim() < info.in_ndim() {
            return Err(Error::msg("not enough free dimensions"));
        }
        let info = info.as_ref();
        let mut n_idx = 0;
        let mut r_idx = 0;
        let mut new_slice = Vec::new();
        let mut new_axes = Vec::new();
        let reader_slice = self.slice.as_slice();
        while (r_idx < reader_slice.len()) & (n_idx < info.len()) {
            let n = &info[n_idx];
            let r = &reader_slice[r_idx];
            let a = &self.axes[r_idx];
            if self.operations.contains_key(a) {
                new_slice.push(*r);
                new_axes.push(*a);
                r_idx += 1;
            } else {
                match (n, r) {
                    (
                        SliceInfoElem::Slice {
                            start: info_start,
                            end: info_end,
                            step: info_step,
                        },
                        SliceInfoElem::Slice { start, end, step },
                    ) => {
                        let new_start = start + info_start;
                        let new_end = match (info_end, end) {
                            (Some(m), Some(n)) => *n.min(&(start + info_step * m)),
                            (None, Some(n)) => *n,
                            _ => panic!("slice has no end"),
                        };
                        let new_step = (step * info_step).abs();
                        new_slice.push(SliceInfoElem::Slice {
                            start: new_start,
                            end: Some(new_end),
                            step: new_step,
                        });
                        new_axes.push(*a);
                        n_idx += 1;
                        r_idx += 1;
                    }
                    (SliceInfoElem::Index(k), SliceInfoElem::Slice { start, end, step }) => {
                        if *k < 0 {
                            new_slice.push(SliceInfoElem::Index(end.unwrap_or(0) + step.abs() * k))
                        } else {
                            new_slice.push(SliceInfoElem::Index(start + step.abs() * k));
                        }
                        new_axes.push(*a);
                        n_idx += 1;
                        r_idx += 1;
                    }
                    (SliceInfoElem::NewAxis, SliceInfoElem::NewAxis) => {
                        new_slice.push(SliceInfoElem::NewAxis);
                        new_slice.push(SliceInfoElem::NewAxis);
                        new_axes.push(Axis::New);
                        new_axes.push(Axis::New);
                        n_idx += 1;
                        r_idx += 1;
                    }
                    (_, SliceInfoElem::NewAxis) => {
                        new_slice.push(SliceInfoElem::NewAxis);
                        new_axes.push(Axis::New);
                        n_idx += 1;
                        r_idx += 1;
                    }
                    (SliceInfoElem::NewAxis, _) => {
                        new_slice.push(SliceInfoElem::NewAxis);
                        new_axes.push(Axis::New);
                        n_idx += 1;
                    }
                    (_, SliceInfoElem::Index(k)) => {
                        new_slice.push(SliceInfoElem::Index(*k));
                        new_axes.push(*a);
                        r_idx += 1;
                    }
                }
            }
        }
        debug_assert_eq!(r_idx, reader_slice.len());
        while n_idx < info.len() {
            debug_assert!(info[n_idx].is_new_axis());
            new_slice.push(SliceInfoElem::NewAxis);
            new_axes.push(Axis::New);
            n_idx += 1;
        }
        Ok(View::new(self.reader.clone(), new_slice, new_axes)
            .with_operations(self.operations.clone()))
    }

    /// slice, but slice elements are in cztyx order, all cztyx must be given,
    /// but axes not present in view will be ignored, view axes are reordered in cztyx order
    pub fn slice_cztyx<I>(&self, info: I) -> Result<View<I::OutDim>>
    where
        I: SliceArg<Ix5>,
    {
        let axes = self.axes();
        let slice: Vec<_> = info
            .as_ref()
            .iter()
            .zip(self.axes.iter())
            .filter_map(|(&s, ax)| if axes.contains(ax) { Some(s) } else { None })
            .collect();
        let new_axes: Vec<_> = [Axis::C, Axis::Z, Axis::T, Axis::Y, Axis::X]
            .into_iter()
            .filter(|ax| axes.contains(ax))
            .collect();
        self.clone()
            .into_dyn()
            .permute_axes(&new_axes)?
            .slice(slice.as_slice())?
            .into_dimensionality()
    }

    /// the pixel intensity at a given index
    pub fn item_at<T>(&self, index: &[isize]) -> Result<T>
    where
        T: Number,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        let slice: Vec<_> = index.iter().map(|s| SliceInfoElem::Index(*s)).collect();
        let view = self.clone().into_dyn().slice(slice.as_slice())?;
        let arr = view.as_array()?;
        Ok(arr.first().unwrap().clone())
    }

    /// collect the view into an ndarray
    pub fn as_array<T>(&self) -> Result<Array<T, D>>
    where
        T: Number,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        Ok(self.as_array_dyn()?.into_dimensionality()?)
    }

    /// collect the view into a dynamic-dimension ndarray
    pub fn as_array_dyn<T>(&self) -> Result<ArrayD<T>>
    where
        T: Number,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        let mut op_xy = IndexMap::new();
        if let Some((&ax, op)) = self.operations.first() {
            if (ax == Axis::X) || (ax == Axis::Y) {
                op_xy.insert(ax, op.clone());
                if let Some((&ax2, op2)) = self.operations.get_index(1) {
                    if (ax2 == Axis::X) || (ax2 == Axis::Y) {
                        op_xy.insert(ax2, op2.clone());
                    }
                }
            }
        }
        let op_czt = if let Some((&ax, op)) = self.operations.get_index(op_xy.len()) {
            IndexMap::from([(ax, op.clone())])
        } else {
            IndexMap::new()
        };
        let mut shape_out = Vec::new();
        let mut slice = Vec::new();
        let mut ax_out = Vec::new();
        for (s, a) in self.slice.iter().zip(&self.axes) {
            match s {
                SliceInfoElem::Slice { start, end, step } => {
                    if let Some(e) = end {
                        if !op_xy.contains_key(a) && !op_czt.contains_key(a) {
                            shape_out.push(((e - start).max(0) / step) as usize);
                            slice.push(SliceInfoElem::Slice {
                                start: 0,
                                end: None,
                                step: 1,
                            });
                            ax_out.push(*a);
                        }
                    } else {
                        panic!("slice has no end")
                    }
                }
                SliceInfoElem::Index(_) => {}
                SliceInfoElem::NewAxis => {
                    shape_out.push(1);
                    slice.push(SliceInfoElem::Index(0));
                    ax_out.push(*a);
                }
            }
        }
        let mut slice_reader = vec![Slice::empty(); 5];
        let mut xy_dim = 0usize;
        let shape = [
            self.size_c as isize,
            self.size_z as isize,
            self.size_t as isize,
            self.size_y as isize,
            self.size_x as isize,
        ];
        for (s, &axis) in self.slice.iter().zip(&self.axes) {
            match axis {
                Axis::New => {}
                _ => match s {
                    SliceInfoElem::Slice { start, end, step } => {
                        if let Axis::X | Axis::Y = axis {
                            if !op_xy.contains_key(&axis) {
                                xy_dim += 1;
                            }
                        }
                        slice_reader[axis as usize] = Slice::new(
                            idx_bnd(*start, shape[axis as usize])?,
                            slc_bnd(end.unwrap(), shape[axis as usize])?,
                            *step,
                        );
                    }
                    SliceInfoElem::Index(j) => {
                        slice_reader[axis as usize] = Slice::new(
                            idx_bnd(*j, shape[axis as usize])?,
                            slc_bnd(*j + 1, shape[axis as usize])?,
                            1,
                        );
                    }
                    SliceInfoElem::NewAxis => panic!("axis cannot be a new axis"),
                },
            }
        }
        let xy = [
            self.slice[Axis::Y.pos(&self.axes, &self.slice)?],
            self.slice[Axis::X.pos(&self.axes, &self.slice)?],
        ];
        let mut array = if let Some((_, op)) = op_czt.first() {
            match op {
                Operation::Max => {
                    ArrayD::<T>::from_elem(shape_out.into_dimension(), T::min_value())
                }
                Operation::Min => {
                    ArrayD::<T>::from_elem(shape_out.into_dimension(), T::max_value())
                }
                _ => ArrayD::<T>::zeros(shape_out.into_dimension()),
            }
        } else {
            ArrayD::<T>::zeros(shape_out.into_dimension())
        };
        let size_c = self.reader.size_c as isize;
        let size_z = self.reader.size_z as isize;
        let size_t = self.reader.size_t as isize;

        let mut axes_out_idx = [None; 5];
        for (i, ax) in ax_out.iter().enumerate() {
            axes_out_idx[*ax as usize] = Some(i);
        }

        for (c, z, t) in iproduct!(&slice_reader[0], &slice_reader[1], &slice_reader[2]) {
            if let Some(i) = axes_out_idx[0] {
                slice[i] = SliceInfoElem::Index(c)
            };
            if let Some(i) = axes_out_idx[1] {
                slice[i] = SliceInfoElem::Index(z)
            };
            if let Some(i) = axes_out_idx[2] {
                slice[i] = SliceInfoElem::Index(t)
            };
            let frame = self.reader.get_frame(
                (c % size_c) as usize,
                (z % size_z) as usize,
                (t % size_t) as usize,
            )?;

            let arr_frame: Array2<T> = frame.try_into()?;
            let arr_frame = match xy_dim {
                0 => {
                    if op_xy.contains_key(&Axis::X) && op_xy.contains_key(&Axis::Y) {
                        let xys = slice_info::<Ix2>(&xy)?;
                        let (&ax0, op0) = op_xy.first().unwrap();
                        let (&ax1, op1) = op_xy.get_index(1).unwrap();
                        let a = arr_frame.slice(xys).to_owned();
                        let b = op0.operate(a, ax0 as usize - 3)?;
                        let c: &Array1<T> = unsafe { transmute(&b) };
                        let d = op1.operate(c.to_owned(), ax1 as usize - 3)?;
                        let e: &Array0<T> = unsafe { transmute(&d) };
                        e.to_owned().into_dyn()
                    } else if op_xy.contains_key(&Axis::X) || op_xy.contains_key(&Axis::Y) {
                        let xys = slice_info::<Ix1>(&xy)?;
                        let (&ax, op) = op_xy.first().unwrap();
                        let a = arr_frame.slice(xys).to_owned();
                        let b = op.operate(a, ax as usize - 3)?;
                        let c: &Array0<T> = unsafe { transmute(&b) };
                        c.to_owned().into_dyn()
                    } else {
                        let xys = slice_info::<Ix0>(&xy)?;
                        arr_frame.slice(xys).to_owned().into_dyn()
                    }
                }
                1 => {
                    if op_xy.contains_key(&Axis::X) || op_xy.contains_key(&Axis::Y) {
                        let xys = slice_info::<Ix2>(&xy)?;
                        let (&ax, op) = op_xy.first().unwrap();
                        let a = arr_frame.slice(xys).to_owned();
                        let b = op.operate(a, ax as usize - 3)?;
                        let c: &Array1<T> = unsafe { transmute(&b) };
                        c.to_owned().into_dyn()
                    } else {
                        let xys = slice_info::<Ix1>(&xy)?;
                        arr_frame.slice(xys).to_owned().into_dyn()
                    }
                }
                2 => {
                    let xys = slice_info::<Ix2>(&xy)?;
                    if axes_out_idx[4] < axes_out_idx[3] {
                        arr_frame.t().slice(xys).to_owned().into_dyn()
                    } else {
                        arr_frame.slice(xys).to_owned().into_dyn()
                    }
                }
                _ => {
                    panic!("xy cannot be 3d or more");
                }
            };
            if let Some((_, op)) = op_czt.first() {
                match op {
                    Operation::Max => {
                        array
                            .slice_mut(slice.as_slice())
                            .zip_mut_with(&arr_frame, |x, y| {
                                *x = if *x >= *y { x.clone() } else { y.clone() }
                            });
                    }
                    Operation::Min => {
                        array
                            .slice_mut(slice.as_slice())
                            .zip_mut_with(&arr_frame, |x, y| {
                                *x = if *x < *y { x.clone() } else { y.clone() }
                            });
                    }
                    Operation::Sum => {
                        array
                            .slice_mut(slice.as_slice())
                            .zip_mut_with(&arr_frame, |x, y| *x += y.clone());
                    }
                    Operation::Mean => {
                        array
                            .slice_mut(slice.as_slice())
                            .zip_mut_with(&arr_frame, |x, y| *x += y.clone());
                    }
                }
            } else {
                array.slice_mut(slice.as_slice()).assign(&arr_frame)
            }
        }
        let mut out = Some(array);
        let ax_out: HashMap<Axis, usize> = ax_out
            .into_iter()
            .enumerate()
            .map(|(i, a)| (a, i))
            .collect();
        for (ax, op) in self.operations.iter().skip(op_xy.len() + op_czt.len()) {
            if let Some(&idx) = ax_out.get(ax) {
                let arr = out.take().unwrap();
                let _ = out.insert(unsafe { transmute_copy(&op.operate(arr, idx)?) });
            }
        }
        let mut n = 1;
        for (&ax, size) in self.axes.iter().zip(self.shape_all().iter()) {
            if let Some(Operation::Mean) = self.operations.get(&ax) {
                if (ax == Axis::C) || (ax == Axis::Z) || (ax == Axis::T) {
                    n *= size;
                }
            }
        }
        let array = if n == 1 {
            out.take().unwrap()
        } else {
            let m = T::from_usize(n).unwrap_or_else(|| T::zero());
            out.take().unwrap().mapv(|x| x / m.clone())
        };
        Ok(array)
    }

    /// retrieve a single frame at czt, sliced accordingly
    pub fn get_frame<T>(&self, c: isize, z: isize, t: isize) -> Result<Array2<T>>
    where
        T: Number,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        self.slice_cztyx(s![c, z, t, .., ..])?.as_array()
    }

    fn get_stat<T>(&self, operation: Operation) -> Result<T>
    where
        T: Number + Sum,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        let arr: ArrayD<T> = self.as_array_dyn()?;
        Ok(match operation {
            Operation::Max => arr
                .flatten()
                .into_iter()
                .reduce(|a, b| if a > b { a } else { b })
                .unwrap_or_else(|| T::min_value()),
            Operation::Min => arr
                .flatten()
                .into_iter()
                .reduce(|a, b| if a < b { a } else { b })
                .unwrap_or_else(|| T::max_value()),
            Operation::Sum => arr.flatten().into_iter().sum(),
            Operation::Mean => {
                arr.flatten().into_iter().sum::<T>()
                    / T::from_usize(arr.len()).ok_or_else(|| {
                        anyhow!("cannot convert {} into {}", arr.len(), type_name::<T>())
                    })?
            }
        })
    }

    /// maximum intensity
    pub fn max<T>(&self) -> Result<T>
    where
        T: Number + Sum,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        self.get_stat(Operation::Max)
    }

    /// minimum intensity
    pub fn min<T>(&self) -> Result<T>
    where
        T: Number + Sum,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        self.get_stat(Operation::Min)
    }

    /// sum intensity
    pub fn sum<T>(&self) -> Result<T>
    where
        T: Number + Sum,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        self.get_stat(Operation::Sum)
    }

    /// mean intensity
    pub fn mean<T>(&self) -> Result<T>
    where
        T: Number + Sum,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        self.get_stat(Operation::Mean)
    }
}

impl<D: Dimension> Deref for View<D> {
    type Target = Reader;

    fn deref(&self) -> &Self::Target {
        self.reader.as_ref()
    }
}

impl<T, D> TryFrom<View<D>> for Array<T, D>
where
    T: Number,
    D: Dimension,
    ArrayD<T>: MinMax,
    Array1<T>: MinMax,
    Array2<T>: MinMax,
{
    type Error = Error;

    fn try_from(view: View<D>) -> Result<Self, Self::Error> {
        view.as_array()
    }
}

impl<T, D> TryFrom<&View<D>> for Array<T, D>
where
    T: Number,
    D: Dimension,
    ArrayD<T>: MinMax,
    Array1<T>: MinMax,
    Array2<T>: MinMax,
{
    type Error = Error;

    fn try_from(view: &View<D>) -> Result<Self, Self::Error> {
        view.as_array()
    }
}

/// trait to define a function to retrieve the only item in a 0d array
pub trait Item {
    fn item<T>(&self) -> Result<T>
    where
        T: Number,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax;
}

impl Item for View<Ix0> {
    fn item<T>(&self) -> Result<T>
    where
        T: Number,
        ArrayD<T>: MinMax,
        Array1<T>: MinMax,
        Array2<T>: MinMax,
    {
        Ok(self
            .as_array()?
            .first()
            .ok_or_else(|| anyhow!("Empty view"))?
            .clone())
    }
}
