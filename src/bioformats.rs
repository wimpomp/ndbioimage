use anyhow::Result;
use j4rs::{Instance, InvocationArg, Jvm, JvmBuilder};
use std::cell::OnceCell;
use std::rc::Rc;

thread_local! {
    static JVM: OnceCell<Rc<Jvm>> = const { OnceCell::new() }
}

/// Ensure 1 jvm per thread
fn jvm() -> Rc<Jvm> {
    JVM.with(|cell| {
        cell.get_or_init(move || {
            #[cfg(feature = "python")]
            let path = crate::py::ndbioimage_file().unwrap();

            #[cfg(not(feature = "python"))]
            let path = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();

            let class_path = if path.join("jassets").exists() {
                path.as_path()
            } else {
                path.parent().unwrap()
            };
            if !class_path.join("jassets").exists() {
                panic!(
                    "jassets directory does not exist in {}",
                    class_path.display()
                );
            }

            Rc::new(
                JvmBuilder::new()
                    .skip_setting_native_lib()
                    .with_base_path(class_path.to_str().unwrap())
                    .build()
                    .expect("Failed to build JVM"),
            )
        })
        .clone()
    })
}

pub fn download_bioformats(gpl_formats: bool) -> Result<()> {
    #[cfg(feature = "python")]
    let path = crate::py::ndbioimage_file()?;

    #[cfg(not(feature = "python"))]
    let path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let class_path = path.parent().unwrap();
    let jassets = class_path.join("jassets");
    if !jassets.exists() {
        std::fs::create_dir_all(jassets)?;
    }
    println!("installing jassets in {}", class_path.display());
    let jvm = JvmBuilder::new()
        .skip_setting_native_lib()
        .with_base_path(class_path.to_str().unwrap())
        .with_maven_settings(j4rs::MavenSettings::new(vec![
            j4rs::MavenArtifactRepo::from(
                "openmicroscopy::https://artifacts.openmicroscopy.org/artifactory/ome.releases",
            ),
        ]))
        .build()?;

    jvm.deploy_artifact(&j4rs::MavenArtifact::from("ome:bioformats_package:8.3.0"))?;

    if gpl_formats {
        jvm.deploy_artifact(&j4rs::MavenArtifact::from("ome:formats-gpl:8.3.0"))?;
    }

    Ok(())
}

macro_rules! method_return {
    ($R:ty$(|c)?) => { Result<$R> };
    () => { Result<()> };
}

macro_rules! method_arg {
    ($n:tt: $t:ty|p) => {
        InvocationArg::try_from($n)?.into_primitive()?
    };
    ($n:tt: $t:ty) => {
        InvocationArg::try_from($n)?
    };
}

macro_rules! method {
    ($name:ident, $method:expr $(,[$($n:tt: $t:ty$(|$p:tt)?),*])? $(=> $tt:ty$(|$c:tt)?)?) => {
        #[allow(dead_code)]
        pub(crate) fn $name(&self, $($($n: $t),*)?) -> method_return!($($tt)?) {
            let args: Vec<InvocationArg> = vec![$($( method_arg!($n:$t$(|$p)?) ),*)?];
            let _result = jvm().invoke(&self.0, $method, &args)?;

            macro_rules! method_result {
                ($R:ty|c) => {
                    Ok(jvm().to_rust(_result)?)
                };
                ($R:ty|d) => {
                    Ok(jvm().to_rust_deserialized(_result)?)
                };
                ($R:ty) => {
                    Ok(_result)
                };
                () => {
                    Ok(())
                };
            }

            method_result!($($tt$(|$c)?)?)
        }
    };
}

fn transmute_vec<T, U>(vec: Vec<T>) -> Vec<U> {
    unsafe {
        // Ensure the original vector is not dropped.
        let mut v_clone = std::mem::ManuallyDrop::new(vec);
        Vec::from_raw_parts(
            v_clone.as_mut_ptr() as *mut U,
            v_clone.len(),
            v_clone.capacity(),
        )
    }
}

/// Wrapper around bioformats java class loci.common.DebugTools
pub struct DebugTools;

impl DebugTools {
    /// set debug root level: ERROR, DEBUG, TRACE, INFO, OFF
    pub fn set_root_level(level: &str) -> Result<()> {
        jvm().invoke_static(
            "loci.common.DebugTools",
            "setRootLevel",
            &[InvocationArg::try_from(level)?],
        )?;
        Ok(())
    }
}

/// Wrapper around bioformats java class loci.formats.ChannelSeparator
pub(crate) struct ChannelSeparator(Instance);

impl ChannelSeparator {
    pub(crate) fn new(image_reader: &ImageReader) -> Result<Self> {
        let jvm = jvm();
        let channel_separator = jvm.create_instance(
            "loci.formats.ChannelSeparator",
            &[InvocationArg::from(jvm.clone_instance(&image_reader.0)?)],
        )?;
        Ok(ChannelSeparator(channel_separator))
    }

    pub(crate) fn open_bytes(&self, index: i32) -> Result<Vec<u8>> {
        Ok(transmute_vec(self.open_bi8(index)?))
    }

    method!(open_bi8, "openBytes", [index: i32|p] => Vec<i8>|c);
    method!(get_index, "getIndex", [z: i32|p, c: i32|p, t: i32|p] => i32|c);
}

/// Wrapper around bioformats java class loci.formats.ImageReader
pub struct ImageReader(Instance);

impl Drop for ImageReader {
    fn drop(&mut self) {
        self.close().unwrap()
    }
}

impl ImageReader {
    pub(crate) fn new() -> Result<Self> {
        let reader = jvm().create_instance("loci.formats.ImageReader", InvocationArg::empty())?;
        Ok(ImageReader(reader))
    }

    pub(crate) fn open_bytes(&self, index: i32) -> Result<Vec<u8>> {
        Ok(transmute_vec(self.open_bi8(index)?))
    }

    pub(crate) fn ome_xml(&self) -> Result<String> {
        let mds = self.get_metadata_store()?;
        Ok(jvm()
            .chain(&mds)?
            .cast("loci.formats.ome.OMEPyramidStore")?
            .invoke("dumpXML", InvocationArg::empty())?
            .to_rust()?)
    }

    method!(set_metadata_store, "setMetadataStore", [ome_data: Instance]);
    method!(get_metadata_store, "getMetadataStore" => Instance);
    method!(set_id, "setId", [id: &str]);
    method!(set_series, "setSeries", [series: i32|p]);
    method!(open_bi8, "openBytes", [index: i32|p] => Vec<i8>|c);
    method!(get_size_x, "getSizeX" => i32|c);
    method!(get_size_y, "getSizeY" => i32|c);
    method!(get_size_c, "getSizeC" => i32|c);
    method!(get_size_t, "getSizeT" => i32|c);
    method!(get_size_z, "getSizeZ" => i32|c);
    method!(get_pixel_type, "getPixelType" => i32|c);
    method!(is_little_endian, "isLittleEndian" => bool|c);
    method!(is_rgb, "isRGB" => bool|c);
    method!(is_interleaved, "isInterleaved" => bool|c);
    method!(get_index, "getIndex", [z: i32|p, c: i32|p, t: i32|p] => i32|c);
    method!(get_rgb_channel_count, "getRGBChannelCount" => i32|c);
    method!(is_indexed, "isIndexed" => bool|c);
    method!(get_8bit_lookup_table, "get8BitLookupTable" => Instance);
    method!(get_16bit_lookup_table, "get16BitLookupTable" => Instance);
    method!(close, "close");
}

/// Wrapper around bioformats java class loci.formats.MetadataTools
pub(crate) struct MetadataTools(Instance);

impl MetadataTools {
    pub(crate) fn new() -> Result<Self> {
        let meta_data_tools =
            jvm().create_instance("loci.formats.MetadataTools", InvocationArg::empty())?;
        Ok(MetadataTools(meta_data_tools))
    }

    method!(create_ome_xml_metadata, "createOMEXMLMetadata" => Instance);
}
