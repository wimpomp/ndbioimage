use crate::axes::Axis;
use crate::bioformats::download_bioformats;
use crate::error::Error;
use crate::metadata::Metadata;
use crate::reader::{PixelType, Reader};
use crate::view::{Item, View};
use itertools::Itertools;
use ndarray::{Ix0, IxDyn, SliceInfoElem};
use numpy::IntoPyArray;
use ome_metadata::Ome;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PyInt, PyList, PySlice, PySliceMethods, PyString, PyTuple};
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};
use std::path::PathBuf;
use std::sync::Arc;

impl From<crate::error::Error> for PyErr {
    fn from(err: crate::error::Error) -> PyErr {
        PyErr::new::<PyValueError, _>(err.to_string())
    }
}

#[pyclass(module = "ndbioimage.ndbioimage_rs")]
struct ViewConstructor;

#[pymethods]
impl ViewConstructor {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }

    #[staticmethod]
    fn __call__(state: String) -> PyResult<PyView> {
        if let Ok(new) = from_str(&state) {
            Ok(new)
        } else {
            Err(PyErr::new::<PyValueError, _>(
                "cannot parse state".to_string(),
            ))
        }
    }
}

#[pyclass(subclass, module = "ndbioimage.ndbioimage_rs")]
#[pyo3(name = "View")]
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PyView {
    view: View<IxDyn>,
    dtype: PixelType,
    ome: Arc<Ome>,
}

#[pymethods]
impl PyView {
    /// new view on a file at path, open series #, open as dtype: (u)int(8/16/32) or float(32/64)
    #[new]
    #[pyo3(signature = (path, series = 0, dtype = "uint16", axes = "cztyx"))]
    fn new<'py>(
        py: Python<'py>,
        path: Bound<'py, PyAny>,
        series: usize,
        dtype: &str,
        axes: &str,
    ) -> PyResult<Self> {
        if path.is_instance_of::<Self>() {
            Ok(path.cast_into::<Self>()?.extract::<Self>()?)
        } else {
            let builtins = PyModule::import(py, "builtins")?;
            let mut path = PathBuf::from(
                builtins
                    .getattr("str")?
                    .call1((path,))?
                    .cast_into::<PyString>()?
                    .extract::<String>()?,
            );
            if path.is_dir() {
                for file in path.read_dir()?.flatten() {
                    let p = file.path();
                    if file.path().is_file() && (p.extension() == Some("tif".as_ref())) {
                        path = p;
                        break;
                    }
                }
            }
            let axes = axes
                .chars()
                .map(|a| a.to_string().parse())
                .collect::<Result<Vec<Axis>, Error>>()?;
            let reader = Reader::new(&path, series)?;
            let view = View::new_with_axes(Arc::new(reader), axes)?;
            let dtype = dtype.parse()?;
            let ome = Arc::new(view.get_ome()?);
            Ok(Self { view, dtype, ome })
        }
    }

    fn squeeze<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let view = self.view.squeeze()?;
        if view.ndim() == 0 {
            Ok(match self.dtype {
                PixelType::I8 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i8>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U8 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u8>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I16 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i16>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U16 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u16>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f64>()?
                    .into_pyobject(py)?
                    .into_any(),
            })
        } else {
            PyView {
                view,
                dtype: self.dtype.clone(),
                ome: self.ome.clone(),
            }
            .into_bound_py_any(py)
        }
    }

    /// close the file: does nothing as this is handled automatically
    fn close(&self) -> PyResult<()> {
        Ok(())
    }

    /// change the data type of the view: (u)int(8/16/32) or float(32/64)
    fn as_type(&self, dtype: &str) -> PyResult<PyView> {
        Ok(PyView {
            view: self.view.clone(),
            dtype: dtype.parse()?,
            ome: self.ome.clone(),
        })
    }

    /// change the data type of the view: (u)int(8/16/32) or float(32/64)
    fn astype(&self, dtype: &str) -> PyResult<PyView> {
        self.as_type(dtype)
    }

    /// slice the view and return a new view or a single number
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        n: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let slice: Vec<_> = if n.is_instance_of::<PyTuple>() {
            n.cast_into::<PyTuple>()?.into_iter().collect()
        } else if n.is_instance_of::<PyList>() {
            n.cast_into::<PyList>()?.into_iter().collect()
        } else {
            vec![n]
        };
        let mut new_slice = Vec::new();
        let mut ellipsis = None;
        let shape = self.view.shape();
        for (i, (s, t)) in slice.iter().zip(shape.iter()).enumerate() {
            if s.is_instance_of::<PyInt>() {
                new_slice.push(SliceInfoElem::Index(s.cast::<PyInt>()?.extract::<isize>()?));
            } else if s.is_instance_of::<PySlice>() {
                let u = s.cast::<PySlice>()?.indices(*t as isize)?;
                new_slice.push(SliceInfoElem::Slice {
                    start: u.start,
                    end: Some(u.stop),
                    step: u.step,
                });
            } else if s.is_instance_of::<PyEllipsis>() {
                if ellipsis.is_some() {
                    return Err(PyErr::new::<PyValueError, _>(
                        "cannot have more than one ellipsis".to_string(),
                    ));
                }
                let _ = ellipsis.insert(i);
            } else {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "cannot convert {:?} to slice",
                    s
                )));
            }
        }
        if new_slice.len() > shape.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "got more indices ({}) than dimensions ({})",
                new_slice.len(),
                shape.len()
            )));
        }
        while new_slice.len() < shape.len() {
            if let Some(i) = ellipsis {
                new_slice.insert(
                    i,
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    },
                )
            } else {
                new_slice.push(SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                })
            }
        }
        let view = self.view.slice(new_slice.as_slice())?;
        if view.ndim() == 0 {
            Ok(match self.dtype {
                PixelType::I8 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i8>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U8 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u8>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I16 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i16>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U16 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u16>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i128>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u128>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f64>()?
                    .into_pyobject(py)?
                    .into_any(),
            })
        } else {
            PyView {
                view,
                dtype: self.dtype.clone(),
                ome: self.ome.clone(),
            }
            .into_bound_py_any(py)
        }
    }

    #[pyo3(signature = (dtype = None))]
    fn __array__<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(dtype) = dtype {
            self.as_type(dtype)?.as_array(py)
        } else {
            self.as_array(py)
        }
    }

    fn __contains__(&self, _item: Bound<PyAny>) -> PyResult<bool> {
        Err(PyNotImplementedError::new_err("contains not implemented"))
    }

    fn __enter__<'py>(slf: PyRef<'py, Self>) -> PyResult<PyRef<'py, Self>> {
        Ok(slf)
    }

    #[allow(unused_variables)]
    #[pyo3(signature = (exc_type=None, exc_val=None, exc_tb=None))]
    fn __exit__(
        &self,
        exc_type: Option<Bound<PyAny>>,
        exc_val: Option<Bound<PyAny>>,
        exc_tb: Option<Bound<PyAny>>,
    ) -> PyResult<()> {
        self.close()
    }

    fn __reduce__(&self) -> PyResult<(ViewConstructor, (String,))> {
        if let Ok(s) = to_string(self) {
            Ok((ViewConstructor, (s,)))
        } else {
            Err(PyErr::new::<PyValueError, _>(
                "cannot get state".to_string(),
            ))
        }
    }

    fn __copy__(&self) -> Self {
        Self {
            view: self.view.clone(),
            dtype: self.dtype.clone(),
            ome: self.ome.clone(),
        }
    }

    fn copy(&self) -> Self {
        Self {
            view: self.view.clone(),
            dtype: self.dtype.clone(),
            ome: self.ome.clone(),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.view.len())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.view.summary()?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.view.path.display().to_string())
    }

    /// retrieve a single frame at czt, sliced accordingly
    fn get_frame<'py>(
        &self,
        py: Python<'py>,
        c: isize,
        z: isize,
        t: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.dtype {
            PixelType::I8 => self
                .view
                .get_frame::<i8, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U8 => self
                .view
                .get_frame::<u8, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::I16 => self
                .view
                .get_frame::<i16, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U16 => self
                .view
                .get_frame::<u16, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::I32 => self
                .view
                .get_frame::<i32, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U32 => self
                .view
                .get_frame::<u32, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::F32 => self
                .view
                .get_frame::<f32, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::F64 => self
                .view
                .get_frame::<f64, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::I64 => self
                .view
                .get_frame::<i64, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U64 => self
                .view
                .get_frame::<u64, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::I128 => self
                .view
                .get_frame::<i64, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U128 => self
                .view
                .get_frame::<u64, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::F128 => self
                .view
                .get_frame::<f64, _>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
        })
    }

    fn flatten<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.dtype {
            PixelType::I8 => self.view.flatten::<i8>()?.into_pyarray(py).into_any(),
            PixelType::U8 => self.view.flatten::<u8>()?.into_pyarray(py).into_any(),
            PixelType::I16 => self.view.flatten::<i16>()?.into_pyarray(py).into_any(),
            PixelType::U16 => self.view.flatten::<u16>()?.into_pyarray(py).into_any(),
            PixelType::I32 => self.view.flatten::<i32>()?.into_pyarray(py).into_any(),
            PixelType::U32 => self.view.flatten::<u32>()?.into_pyarray(py).into_any(),
            PixelType::F32 => self.view.flatten::<f32>()?.into_pyarray(py).into_any(),
            PixelType::F64 => self.view.flatten::<f64>()?.into_pyarray(py).into_any(),
            PixelType::I64 => self.view.flatten::<i64>()?.into_pyarray(py).into_any(),
            PixelType::U64 => self.view.flatten::<u64>()?.into_pyarray(py).into_any(),
            PixelType::I128 => self.view.flatten::<i64>()?.into_pyarray(py).into_any(),
            PixelType::U128 => self.view.flatten::<u64>()?.into_pyarray(py).into_any(),
            PixelType::F128 => self.view.flatten::<f64>()?.into_pyarray(py).into_any(),
        })
    }

    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        Ok(match self.dtype {
            PixelType::I8 => self.view.to_bytes::<i8>()?,
            PixelType::U8 => self.view.to_bytes::<u8>()?,
            PixelType::I16 => self.view.to_bytes::<i16>()?,
            PixelType::U16 => self.view.to_bytes::<u16>()?,
            PixelType::I32 => self.view.to_bytes::<i32>()?,
            PixelType::U32 => self.view.to_bytes::<u32>()?,
            PixelType::F32 => self.view.to_bytes::<f32>()?,
            PixelType::F64 => self.view.to_bytes::<f64>()?,
            PixelType::I64 => self.view.to_bytes::<i64>()?,
            PixelType::U64 => self.view.to_bytes::<u64>()?,
            PixelType::I128 => self.view.to_bytes::<i64>()?,
            PixelType::U128 => self.view.to_bytes::<u64>()?,
            PixelType::F128 => self.view.to_bytes::<f64>()?,
        })
    }

    fn tobytes(&self) -> PyResult<Vec<u8>> {
        self.to_bytes()
    }

    /// retrieve the ome metadata as an XML string
    fn get_ome_xml(&self) -> PyResult<String> {
        Ok(self.view.get_ome_xml()?)
    }

    /// the file path
    #[getter]
    fn path(&self) -> PyResult<String> {
        Ok(self.view.path.display().to_string())
    }

    /// the series in the file
    #[getter]
    fn series(&self) -> PyResult<usize> {
        Ok(self.view.series)
    }

    /// the axes in the view
    #[getter]
    fn axes(&self) -> String {
        self.view.axes().iter().map(|a| format!("{:?}", a)).join("")
    }

    /// the shape of the view
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.view.shape()
    }

    #[getter]
    fn slice(&self) -> PyResult<Vec<String>> {
        Ok(self
            .view
            .get_slice()
            .iter()
            .map(|s| format!("{:#?}", s))
            .collect())
    }

    /// the number of pixels in the view
    #[getter]
    fn size(&self) -> usize {
        self.view.size()
    }

    /// the number of dimensions in the view
    #[getter]
    fn ndim(&self) -> usize {
        self.view.ndim()
    }

    /// find the position of an axis
    #[pyo3(text_signature = "axis: str | int")]
    fn get_ax(&self, axis: Bound<PyAny>) -> PyResult<usize> {
        if axis.is_instance_of::<PyString>() {
            let axis = axis
                .cast_into::<PyString>()?
                .extract::<String>()?
                .parse::<Axis>()?;
            self.view
                .axes()
                .iter()
                .position(|a| *a == axis)
                .ok_or_else(|| {
                    PyErr::new::<PyValueError, _>(format!("cannot find axis {:?}", axis))
                })
        } else if axis.is_instance_of::<PyInt>() {
            Ok(axis.cast_into::<PyInt>()?.extract::<usize>()?)
        } else {
            Err(PyErr::new::<PyValueError, _>(
                "cannot convert to axis".to_string(),
            ))
        }
    }

    /// swap two axes
    #[pyo3(text_signature = "ax0: str | int, ax1: str | int")]
    fn swap_axes(&self, ax0: Bound<PyAny>, ax1: Bound<PyAny>) -> PyResult<Self> {
        let ax0 = self.get_ax(ax0)?;
        let ax1 = self.get_ax(ax1)?;
        let view = self.view.swap_axes(ax0, ax1)?;
        Ok(PyView {
            view,
            dtype: self.dtype.clone(),
            ome: self.ome.clone(),
        })
    }

    /// permute the order of the axes
    #[pyo3(signature = (axes = None), text_signature = "axes: list[str | int] = None")]
    fn transpose(&self, axes: Option<Vec<Bound<PyAny>>>) -> PyResult<Self> {
        let view = if let Some(axes) = axes {
            let ax = axes
                .into_iter()
                .map(|a| self.get_ax(a))
                .collect::<Result<Vec<_>, _>>()?;
            self.view.permute_axes(&ax)?
        } else {
            self.view.transpose()?
        };
        Ok(PyView {
            view,
            dtype: self.dtype.clone(),
            ome: self.ome.clone(),
        })
    }

    #[allow(non_snake_case)]
    #[getter]
    fn T(&self) -> PyResult<Self> {
        self.transpose(None)
    }

    /// collect data into a numpy array
    fn as_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.dtype {
            PixelType::I8 => self.view.as_array_dyn::<i8>()?.into_pyarray(py).into_any(),
            PixelType::U8 => self.view.as_array_dyn::<u8>()?.into_pyarray(py).into_any(),
            PixelType::I16 => self.view.as_array_dyn::<i16>()?.into_pyarray(py).into_any(),
            PixelType::U16 => self.view.as_array_dyn::<u16>()?.into_pyarray(py).into_any(),
            PixelType::I32 => self.view.as_array_dyn::<i32>()?.into_pyarray(py).into_any(),
            PixelType::U32 => self.view.as_array_dyn::<u32>()?.into_pyarray(py).into_any(),
            PixelType::F32 => self.view.as_array_dyn::<f32>()?.into_pyarray(py).into_any(),
            PixelType::F64 => self.view.as_array_dyn::<f64>()?.into_pyarray(py).into_any(),
            PixelType::I64 => self.view.as_array_dyn::<i64>()?.into_pyarray(py).into_any(),
            PixelType::U64 => self.view.as_array_dyn::<u64>()?.into_pyarray(py).into_any(),
            PixelType::I128 => self.view.as_array_dyn::<i64>()?.into_pyarray(py).into_any(),
            PixelType::U128 => self.view.as_array_dyn::<u64>()?.into_pyarray(py).into_any(),
            PixelType::F128 => self.view.as_array_dyn::<f64>()?.into_pyarray(py).into_any(),
        })
    }

    #[getter]
    fn get_dtype(&self) -> PyResult<&str> {
        Ok(match self.dtype {
            PixelType::I8 => "int8",
            PixelType::U8 => "uint8",
            PixelType::I16 => "int16",
            PixelType::U16 => "uint16",
            PixelType::I32 => "int32",
            PixelType::U32 => "uint32",
            PixelType::F32 => "float32",
            PixelType::F64 => "float64",
            PixelType::I64 => "int64",
            PixelType::U64 => "uint64",
            PixelType::I128 => "int128",
            PixelType::U128 => "uint128",
            PixelType::F128 => "float128",
        })
    }

    #[setter]
    fn set_dtype(&mut self, dtype: String) -> PyResult<()> {
        self.dtype = dtype.parse()?;
        Ok(())
    }

    /// get the maximum overall or along a given axis
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (axis=None, dtype=None, out=None, keepdims=false, initial=0, r#where=true), text_signature = "axis: str | int")]
    fn max<'py>(
        &self,
        py: Python<'py>,
        axis: Option<Bound<'py, PyAny>>,
        dtype: Option<Bound<'py, PyAny>>,
        out: Option<Bound<'py, PyAny>>,
        keepdims: bool,
        initial: Option<usize>,
        r#where: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(i) = initial
            && i != 0
        {
            Err(Error::NotImplemented(
                "arguments beyond axis are not implemented".to_string(),
            ))?;
        }
        if dtype.is_some() || out.is_some() || keepdims || !r#where {
            Err(Error::NotImplemented(
                "arguments beyond axis are not implemented".to_string(),
            ))?;
        }
        if let Some(axis) = axis {
            PyView {
                dtype: self.dtype.clone(),
                view: self.view.max_proj(self.get_ax(axis)?)?,
                ome: self.ome.clone(),
            }
            .into_bound_py_any(py)
        } else {
            Ok(match self.dtype {
                PixelType::I8 => self.view.max::<i8>()?.into_pyobject(py)?.into_any(),
                PixelType::U8 => self.view.max::<u8>()?.into_pyobject(py)?.into_any(),
                PixelType::I16 => self.view.max::<i16>()?.into_pyobject(py)?.into_any(),
                PixelType::U16 => self.view.max::<u16>()?.into_pyobject(py)?.into_any(),
                PixelType::I32 => self.view.max::<i32>()?.into_pyobject(py)?.into_any(),
                PixelType::U32 => self.view.max::<u32>()?.into_pyobject(py)?.into_any(),
                PixelType::F32 => self.view.max::<f32>()?.into_pyobject(py)?.into_any(),
                PixelType::F64 => self.view.max::<f64>()?.into_pyobject(py)?.into_any(),
                PixelType::I64 => self.view.max::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U64 => self.view.max::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::I128 => self.view.max::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U128 => self.view.max::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::F128 => self.view.max::<f64>()?.into_pyobject(py)?.into_any(),
            })
        }
    }

    /// get the minimum overall or along a given axis
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (axis=None, dtype=None, out=None, keepdims=false, initial=0, r#where=true), text_signature = "axis: str | int")]
    fn min<'py>(
        &self,
        py: Python<'py>,
        axis: Option<Bound<'py, PyAny>>,
        dtype: Option<Bound<'py, PyAny>>,
        out: Option<Bound<'py, PyAny>>,
        keepdims: bool,
        initial: Option<usize>,
        r#where: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(i) = initial
            && i != 0
        {
            Err(Error::NotImplemented(
                "arguments beyond axis are not implemented".to_string(),
            ))?;
        }
        if dtype.is_some() || out.is_some() || keepdims || !r#where {
            Err(Error::NotImplemented(
                "arguments beyond axis are not implemented".to_string(),
            ))?;
        }
        if let Some(axis) = axis {
            PyView {
                dtype: self.dtype.clone(),
                view: self.view.min_proj(self.get_ax(axis)?)?,
                ome: self.ome.clone(),
            }
            .into_bound_py_any(py)
        } else {
            Ok(match self.dtype {
                PixelType::I8 => self.view.min::<i8>()?.into_pyobject(py)?.into_any(),
                PixelType::U8 => self.view.min::<u8>()?.into_pyobject(py)?.into_any(),
                PixelType::I16 => self.view.min::<i16>()?.into_pyobject(py)?.into_any(),
                PixelType::U16 => self.view.min::<u16>()?.into_pyobject(py)?.into_any(),
                PixelType::I32 => self.view.min::<i32>()?.into_pyobject(py)?.into_any(),
                PixelType::U32 => self.view.min::<u32>()?.into_pyobject(py)?.into_any(),
                PixelType::F32 => self.view.min::<f32>()?.into_pyobject(py)?.into_any(),
                PixelType::F64 => self.view.min::<f64>()?.into_pyobject(py)?.into_any(),
                PixelType::I64 => self.view.min::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U64 => self.view.min::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::I128 => self.view.min::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U128 => self.view.min::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::F128 => self.view.min::<f64>()?.into_pyobject(py)?.into_any(),
            })
        }
    }

    #[pyo3(signature = (axis=None, dtype=None, out=None, keepdims=false, *, r#where=true), text_signature = "axis: str | int")]
    fn mean<'py>(
        &self,
        py: Python<'py>,
        axis: Option<Bound<'py, PyAny>>,
        dtype: Option<Bound<'py, PyAny>>,
        out: Option<Bound<'py, PyAny>>,
        keepdims: bool,
        r#where: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if dtype.is_some() || out.is_some() || keepdims || !r#where {
            Err(Error::NotImplemented(
                "arguments beyond axis are not implemented".to_string(),
            ))?;
        }
        if let Some(axis) = axis {
            let dtype = if let PixelType::F32 = self.dtype {
                PixelType::F32
            } else {
                PixelType::F64
            };
            PyView {
                dtype,
                view: self.view.mean_proj(self.get_ax(axis)?)?,
                ome: self.ome.clone(),
            }
            .into_bound_py_any(py)
        } else {
            Ok(match self.dtype {
                PixelType::F32 => self.view.mean::<f32>()?.into_pyobject(py)?.into_any(),
                _ => self.view.mean::<f64>()?.into_pyobject(py)?.into_any(),
            })
        }
    }

    /// get the sum overall or along a given axis
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (axis=None, dtype=None, out=None, keepdims=false, initial=0, r#where=true), text_signature = "axis: str | int")]
    fn sum<'py>(
        &self,
        py: Python<'py>,
        axis: Option<Bound<'py, PyAny>>,
        dtype: Option<Bound<'py, PyAny>>,
        out: Option<Bound<'py, PyAny>>,
        keepdims: bool,
        initial: Option<usize>,
        r#where: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(i) = initial
            && i != 0
        {
            Err(Error::NotImplemented(
                "arguments beyond axis are not implemented".to_string(),
            ))?;
        }
        if dtype.is_some() || out.is_some() || keepdims || !r#where {
            Err(Error::NotImplemented(
                "arguments beyond axis are not implemented".to_string(),
            ))?;
        }
        let dtype = match self.dtype {
            PixelType::I8 => PixelType::I16,
            PixelType::U8 => PixelType::U16,
            PixelType::I16 => PixelType::I32,
            PixelType::U16 => PixelType::U32,
            PixelType::I32 => PixelType::I64,
            PixelType::U32 => PixelType::U64,
            PixelType::F32 => PixelType::F32,
            PixelType::F64 => PixelType::F64,
            PixelType::I64 => PixelType::I128,
            PixelType::U64 => PixelType::U128,
            PixelType::I128 => PixelType::I128,
            PixelType::U128 => PixelType::U128,
            PixelType::F128 => PixelType::F128,
        };
        if let Some(axis) = axis {
            PyView {
                dtype,
                view: self.view.sum_proj(self.get_ax(axis)?)?,
                ome: self.ome.clone(),
            }
            .into_bound_py_any(py)
        } else {
            Ok(match self.dtype {
                PixelType::F32 => self.view.sum::<f32>()?.into_pyobject(py)?.into_any(),
                PixelType::F64 => self.view.sum::<f64>()?.into_pyobject(py)?.into_any(),
                PixelType::I64 => self.view.sum::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U64 => self.view.sum::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::I128 => self.view.sum::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U128 => self.view.sum::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::F128 => self.view.sum::<f64>()?.into_pyobject(py)?.into_any(),
                _ => self.view.sum::<i64>()?.into_pyobject(py)?.into_any(),
            })
        }
    }

    #[getter]
    fn z_stack(&self) -> PyResult<bool> {
        if let Some(s) = self.view.size_ax(Axis::Z) {
            Ok(s > 1)
        } else {
            Ok(false)
        }
    }

    #[getter]
    fn time_series(&self) -> PyResult<bool> {
        if let Some(s) = self.view.size_ax(Axis::T) {
            Ok(s > 1)
        } else {
            Ok(false)
        }
    }

    #[getter]
    fn pixel_size(&self) -> PyResult<Option<f64>> {
        Ok(self.ome.pixel_size()?)
    }

    #[getter]
    fn delta_z(&self) -> PyResult<Option<f64>> {
        Ok(self.ome.delta_z()?)
    }

    #[getter]
    fn time_interval(&self) -> PyResult<Option<f64>> {
        Ok(self.ome.time_interval()?)
    }

    fn exposure_time(&self, channel: usize) -> PyResult<Option<f64>> {
        Ok(self.ome.exposure_time(channel)?)
    }

    fn binning(&self, channel: usize) -> Option<usize> {
        self.ome.binning(channel)
    }

    fn laser_wavelengths(&self, channel: usize) -> PyResult<Option<f64>> {
        Ok(self.ome.laser_wavelengths(channel)?)
    }

    fn laser_power(&self, channel: usize) -> PyResult<Option<f64>> {
        Ok(self.ome.laser_powers(channel)?)
    }

    #[getter]
    fn objective_name(&self) -> Option<String> {
        self.ome.objective_name()
    }

    #[getter]
    fn magnification(&self) -> Option<f64> {
        self.ome.magnification()
    }

    #[getter]
    fn tube_lens_name(&self) -> Option<String> {
        self.ome.tube_lens_name()
    }

    fn filter_set_name(&self, channel: usize) -> Option<String> {
        self.ome.filter_set_name(channel)
    }

    fn gain(&self, channel: usize) -> Option<f64> {
        self.ome.gain(channel)
    }

    /// gives a helpful summary of the recorded experiment
    fn summary(&self) -> PyResult<String> {
        Ok(self.view.summary()?)
    }
}

pub(crate) fn ndbioimage_file() -> PathBuf {
    let file = Python::attach(|py| {
        py.import("ndbioimage")
            .unwrap()
            .filename()
            .unwrap()
            .to_string()
    });
    PathBuf::from(file)
}

#[pyfunction]
#[pyo3(name = "download_bioformats")]
fn py_download_bioformats(gpl_formats: bool) -> PyResult<()> {
    download_bioformats(gpl_formats)?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "ndbioimage_rs")]
fn ndbioimage_rs(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyView>()?;
    m.add_class::<ViewConstructor>()?;
    m.add_function(wrap_pyfunction!(py_download_bioformats, m)?)?;
    Ok(())
}
