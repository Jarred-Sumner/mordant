//! Stamps `MORDANT_SOURCE_REV` with the git rev the pack is built from
//! (T-328), which `cargo mordant --version` prints. Nothing downstream of
//! the compiled artefact carries this: the `.d` file names the new rev with
//! the `.so` byte-identical and rewrites on every run whether the pin moved
//! or not, the cargo fingerprint JSON is untouched across a pin move, and
//! `strings` on the cdylib carries dependencies' source paths and none of
//! mordant's own — all three measured and refuted (T-328's body, when the
//! pack was a dylint library). The rev has to come from the build, not be
//! read off what the build produced.
//!
//! Mirrors `language/crates/scarlet/build.rs`'s `watch_git_head` /
//! `git_stdout`, the org's existing shape for this exact problem.
//!
//! Also stamps the toolchain the binaries are built with. `mordant-driver`
//! links `librustc_driver` from that toolchain, so it can only run the
//! compiler it was built against: `cargo-mordant` hands cargo that
//! toolchain's `rustc` (`MORDANT_SYSROOT`), and the driver finds the shared
//! library through an rpath into the same sysroot.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    watch_git_head(&manifest);
    let rev = git_rev(&manifest).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=MORDANT_SOURCE_REV={rev}");

    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let sysroot = stdout(Command::new(&rustc).args(["--print", "sysroot"]))
        .expect("`rustc --print sysroot` names the toolchain building mordant");
    let version = stdout(Command::new(&rustc).arg("-V")).expect("`rustc -V` names its own version");
    println!("cargo:rustc-env=MORDANT_SYSROOT={sysroot}");
    println!("cargo:rustc-env=MORDANT_RUSTC_VERSION={version}");
    if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{sysroot}/lib");
    }
}

fn stdout(command: &mut Command) -> Option<String> {
    let out = command.output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()
        .map(|s| s.trim().to_string())
}

/// `cargo:rerun-if-changed` on `.git/HEAD` and whatever ref it points at, so
/// a new commit re-runs this script instead of caching a stale rev. Handles
/// a linked worktree, where `.git` is a file naming the real gitdir, not a
/// directory.
fn watch_git_head(repo: &Path) {
    let git = repo.join(".git");
    println!("cargo:rerun-if-changed={}", git.display());
    let head = if git.is_dir() {
        git.join("HEAD")
    } else {
        let Ok(contents) = std::fs::read_to_string(&git) else {
            return;
        };
        let Some(dir) = contents.strip_prefix("gitdir:") else {
            return;
        };
        PathBuf::from(dir.trim()).join("HEAD")
    };
    println!("cargo:rerun-if-changed={}", head.display());
    if let Ok(contents) = std::fs::read_to_string(&head)
        && let Some(r) = contents.strip_prefix("ref: ")
    {
        println!(
            "cargo:rerun-if-changed={}",
            head.parent()
                .unwrap_or(Path::new("."))
                .join(r.trim())
                .display()
        );
    }
}

/// `None` when `git` is unavailable or the tree has no history (a source
/// tarball with `.git` stripped): the build must not fail over metadata
/// that is not essential to compiling the pack.
fn git_rev(repo: &Path) -> Option<String> {
    stdout(
        Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(["rev-parse", "HEAD"]),
    )
}
