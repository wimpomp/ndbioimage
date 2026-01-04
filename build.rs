#[cfg(not(feature = "python"))]
use j4rs::{JvmBuilder, MavenArtifact, MavenArtifactRepo, MavenSettings, errors::J4RsError};
#[cfg(not(feature = "python"))]
use retry::{delay, delay::Exponential, retry};
use std::error::Error;
#[cfg(not(feature = "python"))]
use std::fmt::Display;
#[cfg(not(feature = "python"))]
use std::fmt::Formatter;
#[cfg(not(feature = "python"))]
use std::path::PathBuf;
#[cfg(not(feature = "python"))]
use std::{env, fs};

#[cfg(feature = "python")]
use j4rs::Jvm;

#[cfg(feature = "movie")]
use ffmpeg_sidecar::download::auto_download;

#[cfg(not(feature = "python"))]
#[derive(Clone, Debug)]
enum BuildError {
    BioFormatsNotDownloaded,
}

#[cfg(not(feature = "python"))]
impl Display for BuildError {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(fmt, "Bioformats package not downloaded")
    }
}

#[cfg(not(feature = "python"))]
impl Error for BuildError {}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-changed=build.rs");

    if std::env::var("DOCS_RS").is_err() {
        #[cfg(feature = "movie")]
        auto_download()?;

        #[cfg(not(feature = "python"))]
        {
            retry(
                Exponential::from_millis(1000).map(delay::jitter).take(4),
                deploy_java_artifacts,
            )?;
            let path = default_jassets_path()?;
            if !path.join("bioformats_package-8.3.0.jar").exists() {
                Err(BuildError::BioFormatsNotDownloaded)?;
            }
        }

        #[cfg(feature = "python")]
        {
            let py_src_path = std::env::current_dir()?.join("py").join("ndbioimage");
            let py_jassets_path = py_src_path.join("jassets");
            let py_deps_path = py_src_path.join("deps");
            if py_jassets_path.exists() {
                std::fs::remove_dir_all(&py_jassets_path)?;
            }
            if py_deps_path.exists() {
                std::fs::remove_dir_all(&py_deps_path)?;
            }

            Jvm::copy_j4rs_libs_under(py_src_path.to_str().unwrap())?;

            // rename else maturin will ignore them
            for file in std::fs::read_dir(&py_deps_path)? {
                let f = file?.path().to_str().unwrap().to_string();
                if !f.ends_with("_") {
                    std::fs::rename(&f, std::format!("{f}_"))?;
                }
            }

            // remove so we don't include too much accidentally
            for file in std::fs::read_dir(&py_jassets_path)? {
                let f = file?.path();
                if !f.file_name().unwrap().to_str().unwrap().starts_with("j4rs") {
                    std::fs::remove_file(&f)?;
                }
            }
        }
    }

    Ok(())
}

#[cfg(not(feature = "python"))]
fn default_jassets_path() -> Result<PathBuf, J4RsError> {
    let is_build_script = env::var("OUT_DIR").is_ok();

    let mut start_path = if is_build_script {
        PathBuf::from(env::var("OUT_DIR")?)
    } else {
        env::current_exe()?
    };
    start_path = fs::canonicalize(start_path)?;

    while start_path.pop() {
        for entry in std::fs::read_dir(&start_path)? {
            let path = entry?.path();
            if path.file_name().map(|x| x == "jassets").unwrap_or(false) {
                return Ok(path);
            }
        }
    }

    Err(J4RsError::GeneralError(
        "Can not find jassets directory".to_owned(),
    ))
}

#[cfg(not(feature = "python"))]
fn deploy_java_artifacts() -> Result<(), J4RsError> {
    let jvm = JvmBuilder::new()
        .skip_setting_native_lib()
        .with_maven_settings(MavenSettings::new(vec![MavenArtifactRepo::from(
            "openmicroscopy::https://artifacts.openmicroscopy.org/artifactory/ome.releases",
        )]))
        .build()?;

    jvm.deploy_artifact(&MavenArtifact::from("ome:bioformats_package:8.3.0"))?;

    #[cfg(feature = "gpl-formats")]
    jvm.deploy_artifact(&MavenArtifact::from("ome:formats-gpl:8.3.0"))?;

    Ok(())
}
