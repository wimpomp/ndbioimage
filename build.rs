#[cfg(not(feature = "python"))]
use j4rs::{errors::J4RsError, JvmBuilder, MavenArtifact, MavenArtifactRepo, MavenSettings};

#[cfg(not(feature = "python"))]
use retry::{delay, delay::Exponential, retry};

#[cfg(feature = "python")]
use j4rs::Jvm;

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(not(feature = "python"))]
    if std::env::var("DOCS_RS").is_err() {
        retry(
            Exponential::from_millis(1000).map(delay::jitter).take(4),
            deploy_java_artifacts,
        )?
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

    Ok(())
}

#[cfg(not(feature = "python"))]
fn deploy_java_artifacts() -> Result<(), J4RsError> {
    let jvm = JvmBuilder::new()
        .with_maven_settings(MavenSettings::new(vec![MavenArtifactRepo::from(
            "openmicroscopy::https://artifacts.openmicroscopy.org/artifactory/ome.releases",
        )]))
        .build()?;

    jvm.deploy_artifact(&MavenArtifact::from("ome:bioformats_package:8.1.0"))?;

    #[cfg(feature = "gpl-formats")]
    jvm.deploy_artifact(&MavenArtifact::from("ome:formats-gpl:8.1.0"))?;

    Ok(())
}
