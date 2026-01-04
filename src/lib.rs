#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod bioformats;

pub mod axes;
pub mod metadata;
#[cfg(feature = "python")]
mod py;
pub mod reader;
pub mod stats;
pub mod view;

pub mod colors;
pub mod error;
#[cfg(feature = "movie")]
pub mod movie;
#[cfg(feature = "tiff")]
pub mod tiff;

pub use bioformats::download_bioformats;

#[cfg(test)]
mod tests {
    use crate::axes::Axis;
    use crate::error::Error;
    use crate::reader::{Frame, Reader};
    use crate::stats::MinMax;
    use crate::view::Item;
    use ndarray::{Array, Array4, Array5, NewAxis};
    use ndarray::{Array2, s};
    use rayon::prelude::*;

    fn open(file: &str) -> Result<Reader, Error> {
        let path = std::env::current_dir()?
            .join("tests")
            .join("files")
            .join(file);
        Reader::new(&path, 0)
    }

    fn get_pixel_type(file: &str) -> Result<String, Error> {
        let reader = open(file)?;
        Ok(format!(
            "file: {}, pixel type: {:?}",
            file, reader.pixel_type
        ))
    }

    fn get_frame(file: &str) -> Result<Frame, Error> {
        let reader = open(file)?;
        reader.get_frame(0, 0, 0)
    }

    #[test]
    fn read_ser() -> Result<(), Error> {
        let file = "Experiment-2029.czi";
        let reader = open(file)?;
        println!("size: {}, {}", reader.size_y, reader.size_y);
        let frame = reader.get_frame(0, 0, 0)?;
        if let Ok(arr) = <Frame as TryInto<Array2<i8>>>::try_into(frame) {
            println!("{:?}", arr);
        } else {
            println!("could not convert Frame to Array<i8>");
        }
        Ok(())
    }

    #[test]
    fn read_par() -> Result<(), Error> {
        let files = vec!["Experiment-2029.czi", "test.tif"];
        let pixel_type = files
            .into_par_iter()
            .map(|file| get_pixel_type(file).unwrap())
            .collect::<Vec<_>>();
        println!("{:?}", pixel_type);
        Ok(())
    }

    #[test]
    fn read_frame_par() -> Result<(), Error> {
        let files = vec!["Experiment-2029.czi", "test.tif"];
        let frames = files
            .into_par_iter()
            .map(|file| get_frame(file).unwrap())
            .collect::<Vec<_>>();
        println!("{:?}", frames);
        Ok(())
    }

    #[test]
    fn read_sequence() -> Result<(), Error> {
        let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
        let reader = open(file)?;
        println!("reader: {:?}", reader);
        let frame = reader.get_frame(0, 4, 0)?;
        println!("frame: {:?}", frame);
        let frame = reader.get_frame(0, 2, 0)?;
        println!("frame: {:?}", frame);
        Ok(())
    }

    #[test]
    fn read_sequence1() -> Result<(), Error> {
        let file = "4-Pos_001_002/img_000000000_Cy3-Cy3_filter_000.tif";
        let reader = open(file)?;
        println!("reader: {:?}", reader);
        Ok(())
    }

    #[test]
    fn ome_xml() -> Result<(), Error> {
        let file = "Experiment-2029.czi";
        let reader = open(file)?;
        let xml = reader.get_ome_xml()?;
        println!("{}", xml);
        Ok(())
    }

    #[test]
    fn view() -> Result<(), Error> {
        let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
        let reader = open(file)?;
        let view = reader.view();
        let a = view.slice(s![0, 5, 0, .., ..])?;
        let b = reader.get_frame(0, 5, 0)?;
        let c: Array2<isize> = a.try_into()?;
        let d: Array2<isize> = b.try_into()?;
        assert_eq!(c, d);
        Ok(())
    }

    #[test]
    fn view_shape() -> Result<(), Error> {
        let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
        let reader = open(file)?;
        let view = reader.view();
        let a = view.slice(s![0, ..5, 0, .., 100..200])?;
        let shape = a.shape();
        assert_eq!(shape, vec![5, 1024, 100]);
        Ok(())
    }

    #[test]
    fn view_new_axis() -> Result<(), Error> {
        let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
        let reader = open(file)?;
        let view = reader.view();
        let a = Array5::<u8>::zeros((1, 9, 1, 1024, 1024));
        let a = a.slice(s![0, ..5, 0, NewAxis, 100..200, ..]);
        let v = view.slice(s![0, ..5, 0, NewAxis, 100..200, ..])?;
        assert_eq!(v.shape(), a.shape());
        let a = a.slice(s![NewAxis, .., .., NewAxis, .., .., NewAxis]);
        let v = v.slice(s![NewAxis, .., .., NewAxis, .., .., NewAxis])?;
        assert_eq!(v.shape(), a.shape());
        Ok(())
    }

    #[test]
    fn view_permute_axes() -> Result<(), Error> {
        let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
        let reader = open(file)?;
        let view = reader.view();
        let s = view.shape();
        let mut a = Array5::<u8>::zeros((s[0], s[1], s[2], s[3], s[4]));
        assert_eq!(view.shape(), a.shape());
        let b: Array5<usize> = view.clone().try_into()?;
        assert_eq!(b.shape(), a.shape());

        let view = view.swap_axes(Axis::C, Axis::Z)?;
        a.swap_axes(0, 1);
        assert_eq!(view.shape(), a.shape());
        let b: Array5<usize> = view.clone().try_into()?;
        assert_eq!(b.shape(), a.shape());
        let view = view.permute_axes(&[Axis::X, Axis::Z, Axis::Y])?;
        let a = a.permuted_axes([4, 1, 2, 0, 3]);
        assert_eq!(view.shape(), a.shape());
        let b: Array5<usize> = view.clone().try_into()?;
        assert_eq!(b.shape(), a.shape());
        Ok(())
    }

    macro_rules! test_max {
        ($($name:ident: $b:expr $(,)?)*) => {
            $(
                #[test]
                fn $name() -> Result<(), Error> {
                    let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
                    let reader = open(file)?;
                    let view = reader.view();
                    let array: Array5<usize> = view.clone().try_into()?;
                    let view = view.max_proj($b)?;
                    let a: Array4<usize> = view.clone().try_into()?;
                    let b = array.max($b)?;
                    assert_eq!(a.shape(), b.shape());
                    assert_eq!(a, b);
                    Ok(())
                }
            )*
        };
    }

    test_max! {
        max_c: 0
        max_z: 1
        max_t: 2
        max_y: 3
        max_x: 4
    }

    macro_rules! test_index {
        ($($name:ident: $b:expr $(,)?)*) => {
            $(
                #[test]
                fn $name() -> Result<(), Error> {
                    let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
                    let reader = open(file)?;
                    let view = reader.view();
                    let v4: Array<usize, _> = view.slice($b)?.try_into()?;
                    let a5: Array5<usize> = reader.view().try_into()?;
                    let a4 = a5.slice($b).to_owned();
                    assert_eq!(a4, v4);
                    Ok(())
                }
            )*
        };
    }

    test_index! {
        index_0: s![.., .., .., .., ..]
        index_1: s![0, .., .., .., ..]
        index_2: s![.., 0, .., .., ..]
        index_3: s![.., .., 0, .., ..]
        index_4: s![.., .., .., 0, ..]
        index_5: s![.., .., .., .., 0]
        index_6: s![0, 0, .., .., ..]
        index_7: s![0, .., 0, .., ..]
        index_8: s![0, .., .., 0, ..]
        index_9: s![0, .., .., .., 0]
        index_a: s![.., 0, 0, .., ..]
        index_b: s![.., 0, .., 0, ..]
        index_c: s![.., 0, .., .., 0]
        index_d: s![.., .., 0, 0, ..]
        index_e: s![.., .., 0, .., 0]
        index_f: s![.., .., .., 0, 0]
        index_g: s![0, 0, 0, .., ..]
        index_h: s![0, 0, .., 0, ..]
        index_i: s![0, 0, .., .., 0]
        index_j: s![0, .., 0, 0, ..]
        index_k: s![0, .., 0, .., 0]
        index_l: s![0, .., .., 0, 0]
        index_m: s![0, 0, 0, 0, ..]
        index_n: s![0, 0, 0, .., 0]
        index_o: s![0, 0, .., 0, 0]
        index_p: s![0, .., 0, 0, 0]
        index_q: s![.., 0, 0, 0, 0]
        index_r: s![0, 0, 0, 0, 0]
    }

    #[test]
    fn dyn_view() -> Result<(), Error> {
        let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
        let reader = open(file)?;
        let a = reader.view().into_dyn();
        let b = a.max_proj(1)?;
        let c = b.slice(s![0, 0, .., ..])?;
        let d = c.as_array::<usize>()?;
        assert_eq!(d.shape(), [1024, 1024]);
        Ok(())
    }

    #[test]
    fn item() -> Result<(), Error> {
        let file = "1xp53-01-AP1.czi";
        let reader = open(file)?;
        let view = reader.view();
        let a = view.slice(s![.., 0, 0, 0, 0])?;
        let b = a.slice(s![0])?;
        let item = b.item::<usize>()?;
        assert_eq!(item, 2);
        Ok(())
    }

    #[test]
    fn slice_cztyx() -> Result<(), Error> {
        let file = "1xp53-01-AP1.czi";
        let reader = open(file)?;
        let view = reader.view().max_proj(Axis::Z)?.into_dyn();
        println!("view.axes: {:?}", view.get_axes());
        println!("view.slice: {:?}", view.get_slice());
        let r = view.reset_axes()?;
        println!("r.axes: {:?}", r.get_axes());
        println!("r.slice: {:?}", r.get_slice());
        let a = view.slice_cztyx(s![0, 0, 0, .., ..])?;
        println!("a.axes: {:?}", a.get_axes());
        println!("a.slice: {:?}", a.get_slice());
        assert_eq!(a.axes(), [Axis::Y, Axis::X]);
        Ok(())
    }

    #[test]
    fn reset_axes() -> Result<(), Error> {
        let file = "1xp53-01-AP1.czi";
        let reader = open(file)?;
        let view = reader.view().max_proj(Axis::Z)?;
        let view = view.reset_axes()?;
        assert_eq!(view.axes(), [Axis::C, Axis::New, Axis::T, Axis::Y, Axis::X]);
        let a = view.as_array::<f64>()?;
        assert_eq!(a.ndim(), 5);
        Ok(())
    }

    #[test]
    fn reset_axes2() -> Result<(), Error> {
        let file = "Experiment-2029.czi";
        let reader = open(file)?;
        let view = reader.view().squeeze()?;
        let a = view.reset_axes()?;
        assert_eq!(a.axes(), [Axis::C, Axis::Z, Axis::T, Axis::Y, Axis::X]);
        Ok(())
    }

    #[test]
    fn reset_axes3() -> Result<(), Error> {
        let file = "Experiment-2029.czi";
        let reader = open(file)?;
        let view4 = reader.view().squeeze()?;
        let view = view4.max_proj(Axis::Z)?.into_dyn();
        let slice = view.slice_cztyx(s![0, .., .., .., ..])?.into_dyn();
        let a = slice.as_array::<u16>()?;
        assert_eq!(slice.shape(), [1, 10, 1280, 1280]);
        assert_eq!(a.shape(), [1, 10, 1280, 1280]);
        let r = slice.reset_axes()?;
        let b = r.as_array::<u16>()?;
        assert_eq!(r.shape(), [1, 1, 10, 1280, 1280]);
        assert_eq!(b.shape(), [1, 1, 10, 1280, 1280]);
        let q = slice.max_proj(Axis::C)?.max_proj(Axis::T)?;
        let c = q.as_array::<f64>()?;
        assert_eq!(q.shape(), [1, 1280, 1280]);
        assert_eq!(c.shape(), [1, 1280, 1280]);
        let p = q.reset_axes()?;
        let d = p.as_array::<u16>()?;
        println!("axes: {:?}", p.get_axes());
        println!("operations: {:?}", p.get_operations());
        println!("slice: {:?}", p.get_slice());
        assert_eq!(p.shape(), [1, 1, 1, 1280, 1280]);
        assert_eq!(d.shape(), [1, 1, 1, 1280, 1280]);
        Ok(())
    }

    #[test]
    fn max() -> Result<(), Error> {
        let file = "Experiment-2029.czi";
        let reader = open(file)?;
        let view = reader.view();
        let m = view.max_proj(Axis::T)?;
        let a = m.as_array::<u16>()?;
        assert_eq!(m.shape(), [2, 1, 1280, 1280]);
        assert_eq!(a.shape(), [2, 1, 1280, 1280]);
        let mc = view.max_proj(Axis::C)?;
        let a = mc.as_array::<u16>()?;
        assert_eq!(mc.shape(), [1, 10, 1280, 1280]);
        assert_eq!(a.shape(), [1, 10, 1280, 1280]);
        let mz = mc.max_proj(Axis::Z)?;
        let a = mz.as_array::<u16>()?;
        assert_eq!(mz.shape(), [10, 1280, 1280]);
        assert_eq!(a.shape(), [10, 1280, 1280]);
        let mt = mz.max_proj(Axis::T)?;
        let a = mt.as_array::<u16>()?;
        assert_eq!(mt.shape(), [1280, 1280]);
        assert_eq!(a.shape(), [1280, 1280]);
        Ok(())
    }
}
