use crate::error::Error;
use crate::stats::MinMax;
use ndarray::{Array, Dimension, Ix2, SliceInfo, SliceInfoElem};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{DeserializeAs, SerializeAs};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::str::FromStr;

/// a trait to find axis indices from any object
pub trait Ax {
    /// C: 0, Z: 1, T: 2, Y: 3, X: 4
    fn n(&self) -> usize;

    /// the indices of axes in self.axes, which always has all of CZTYX
    fn pos(&self, axes: &[Axis], slice: &[SliceInfoElem]) -> Result<usize, Error>;

    /// the indices of axes in self.axes, which always has all of CZTYX, but skip axes with an operation
    fn pos_op(
        &self,
        axes: &[Axis],
        slice: &[SliceInfoElem],
        op_axes: &[Axis],
    ) -> Result<usize, Error>;
}

/// Enum for CZTYX axes or a new axis
#[derive(Clone, Copy, Debug, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Axis {
    C,
    Z,
    T,
    Y,
    X,
    New,
}

impl Hash for Axis {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (*self as usize).hash(state);
    }
}

impl FromStr for Axis {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "C" => Ok(Axis::C),
            "Z" => Ok(Axis::Z),
            "T" => Ok(Axis::T),
            "Y" => Ok(Axis::Y),
            "X" => Ok(Axis::X),
            "NEW" => Ok(Axis::New),
            _ => Err(Error::InvalidAxis(s.to_string())),
        }
    }
}

impl Display for Axis {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Axis::C => "C",
            Axis::Z => "Z",
            Axis::T => "T",
            Axis::Y => "Y",
            Axis::X => "X",
            Axis::New => "N",
        };
        write!(f, "{}", s)
    }
}

impl Ax for Axis {
    fn n(&self) -> usize {
        *self as usize
    }

    fn pos(&self, axes: &[Axis], _slice: &[SliceInfoElem]) -> Result<usize, Error> {
        if let Some(pos) = axes.iter().position(|a| a == self) {
            Ok(pos)
        } else {
            Err(Error::AxisNotFound(
                format!("{:?}", self),
                format!("{:?}", axes),
            ))
        }
    }

    fn pos_op(
        &self,
        axes: &[Axis],
        _slice: &[SliceInfoElem],
        _op_axes: &[Axis],
    ) -> Result<usize, Error> {
        self.pos(axes, _slice)
    }
}

impl Ax for usize {
    fn n(&self) -> usize {
        *self
    }

    fn pos(&self, _axes: &[Axis], slice: &[SliceInfoElem]) -> Result<usize, Error> {
        let idx: Vec<_> = slice
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.is_index() { None } else { Some(i) })
            .collect();
        Ok(idx[*self])
    }

    fn pos_op(
        &self,
        axes: &[Axis],
        slice: &[SliceInfoElem],
        op_axes: &[Axis],
    ) -> Result<usize, Error> {
        let idx: Vec<_> = axes
            .iter()
            .zip(slice.iter())
            .enumerate()
            .filter_map(|(i, (ax, s))| {
                if s.is_index() | op_axes.contains(ax) {
                    None
                } else {
                    Some(i)
                }
            })
            .collect();
        debug_assert!(*self < idx.len(), "self: {}, idx: {:?}", self, idx);
        Ok(idx[*self])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum Operation {
    Max,
    Min,
    Sum,
    Mean,
}

impl Operation {
    pub(crate) fn operate<T, D>(
        &self,
        array: Array<T, D>,
        axis: usize,
    ) -> Result<<Array<T, D> as MinMax>::Output, Error>
    where
        D: Dimension,
        Array<T, D>: MinMax,
    {
        match self {
            Operation::Max => array.max(axis),
            Operation::Min => array.min(axis),
            Operation::Sum => array.sum(axis),
            Operation::Mean => array.mean(axis),
        }
    }
}

impl PartialEq for Axis {
    fn eq(&self, other: &Self) -> bool {
        (*self as u8) == (*other as u8)
    }
}

pub(crate) fn slice_info<D: Dimension>(
    info: &[SliceInfoElem],
) -> Result<SliceInfo<&[SliceInfoElem], Ix2, D>, Error> {
    match info.try_into() {
        Ok(slice) => Ok(slice),
        Err(err) => Err(Error::TryInto(err.to_string())),
    }
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "SliceInfoElem")]
pub(crate) enum SliceInfoElemDef {
    Slice {
        start: isize,
        end: Option<isize>,
        step: isize,
    },
    Index(isize),
    NewAxis,
}

impl SerializeAs<SliceInfoElem> for SliceInfoElemDef {
    fn serialize_as<S>(source: &SliceInfoElem, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SliceInfoElemDef::serialize(source, serializer)
    }
}

impl<'de> DeserializeAs<'de, SliceInfoElem> for SliceInfoElemDef {
    fn deserialize_as<D>(deserializer: D) -> Result<SliceInfoElem, D::Error>
    where
        D: Deserializer<'de>,
    {
        SliceInfoElemDef::deserialize(deserializer)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Slice {
    start: isize,
    end: isize,
    step: isize,
}

impl Slice {
    pub(crate) fn new(start: isize, end: isize, step: isize) -> Self {
        Self { start, end, step }
    }

    pub(crate) fn empty() -> Self {
        Self {
            start: 0,
            end: 0,
            step: 1,
        }
    }
}

impl Iterator for Slice {
    type Item = isize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.end - self.start >= self.step {
            let r = self.start;
            self.start += self.step;
            Some(r)
        } else {
            None
        }
    }
}

impl IntoIterator for &Slice {
    type Item = isize;
    type IntoIter = Slice;

    fn into_iter(self) -> Self::IntoIter {
        self.clone()
    }
}
