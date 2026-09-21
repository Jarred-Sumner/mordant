//! `cargo mordant`: `cargo check` with `mordant-driver` compiling the
//! workspace's own crates, so mordant's lints run over them. Dependencies
//! build with plain rustc from the toolchain the driver was built with,
//! since the driver can only read metadata that compiler wrote. All of it
//! goes to `<target>/mordant/check`, so a run leaves the workspace's usual
//! builds alone.
//!
//! This binary does not link the compiler, so it starts, and can say what is
//! missing, where the toolchain it was built with is gone.

use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt::Display;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};

#[path = "../protocol.rs"]
mod protocol;

const USAGE: &str = "\
Run mordant's lints over a cargo workspace.

Usage: cargo mordant [--fix] [<cargo check options>]

Options:
      --fix      Run `cargo fix` instead of `cargo check`, applying the fixes
                 the lints suggest
      --list     Print every lint, with its default level and description
  -V, --version  Print the version, the commit it was built from, and its rustc
  -h, --help     Print this help

Every other option goes to `cargo check` as written: `--workspace`,
`--all-targets`, `-p <package>`, `--keep-going` and so on.

The configuration is the `[mordant]` table of `dylint.toml` in the workspace
root, or the text of MORDANT_TOML when that is set. MORDANT_RUSTFLAGS adds
rustc flags for the linted crates only, as in MORDANT_RUSTFLAGS=\"-D warnings\".";

#[derive(serde::Deserialize)]
struct Metadata {
    workspace_root: PathBuf,
    target_directory: PathBuf,
}

fn main() -> ExitCode {
    let mut args: Vec<OsString> = env::args_os().skip(1).collect();
    // Cargo runs `cargo mordant ..` as `cargo-mordant mordant ..`.
    if args.first().is_some_and(|a| a == "mordant") {
        args.remove(0);
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!(
            "mordant {} ({}, {})",
            env!("CARGO_PKG_VERSION"),
            env!("MORDANT_SOURCE_REV"),
            env!("MORDANT_RUSTC_VERSION")
        );
        return ExitCode::SUCCESS;
    }
    let before = args.len();
    args.retain(|a| a != "--fix");
    let subcommand = if args.len() < before { "fix" } else { "check" };

    let Ok(exe) = env::current_exe() else {
        return fail("could not find this executable's path");
    };
    let driver = exe.with_file_name(format!("mordant-driver{}", env::consts::EXE_SUFFIX));
    if !driver.is_file() {
        return fail(format_args!(
            "no `mordant-driver` beside {}; install both binaries from one build",
            exe.display()
        ));
    }
    let sysroot = Path::new(env!("MORDANT_SYSROOT"));
    let rustc = sysroot
        .join("bin")
        .join(format!("rustc{}", env::consts::EXE_SUFFIX));
    if !rustc.is_file() {
        return fail(format_args!(
            "mordant was built with the toolchain at {}, which is not installed any more; \
             install it again, or rebuild mordant",
            sysroot.display()
        ));
    }
    if args.iter().any(|a| a == "--list") {
        return exit_code(
            Command::new(&driver).arg(protocol::LIST_ARG).status(),
            "mordant-driver",
        );
    }

    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let Some(meta) = metadata(&cargo, option_value(&args, "--manifest-path")) else {
        return fail("could not read the workspace from `cargo metadata`");
    };
    let mut command = Command::new(&cargo);
    command.arg(subcommand);
    if option_value(&args, "--target-dir").is_none() {
        command
            .arg("--target-dir")
            .arg(meta.target_directory.join("mordant").join("check"));
    }
    command
        .args(&args)
        .env("RUSTC", &rustc)
        .env("RUSTC_WORKSPACE_WRAPPER", &driver);
    if env::var_os(protocol::CONFIG_ENV).is_none() {
        let path = meta.workspace_root.join("dylint.toml");
        match fs::read_to_string(&path) {
            Ok(text) => {
                command.env(protocol::CONFIG_ENV, text);
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return fail(format_args!("could not read {}: {err}", path.display())),
        }
    }
    exit_code(command.status(), "cargo")
}

/// The exit status of the program this run hands over to, as its own.
fn exit_code(status: io::Result<ExitStatus>, program: &str) -> ExitCode {
    match status {
        Ok(status) => ExitCode::from(
            status
                .code()
                .and_then(|code| u8::try_from(code).ok())
                .unwrap_or(1),
        ),
        Err(err) => fail(format_args!("could not run {program}: {err}")),
    }
}

fn fail(message: impl Display) -> ExitCode {
    eprintln!("error: mordant: {message}");
    ExitCode::FAILURE
}

/// The workspace `cargo check` will build: the one `--manifest-path` names,
/// or else the one around the current directory. When cargo fails, its own
/// error is printed first.
fn metadata(cargo: &OsStr, manifest_path: Option<&OsStr>) -> Option<Metadata> {
    let mut command = Command::new(cargo);
    command.args(["metadata", "--no-deps", "--format-version", "1"]);
    if let Some(path) = manifest_path {
        command.arg("--manifest-path").arg(path);
    }
    let out = command.output().ok()?;
    if !out.status.success() {
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

/// The value of `--name <value>` or `--name=<value>` in cargo's arguments.
fn option_value<'a>(args: &'a [OsString], name: &str) -> Option<&'a OsStr> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next().map(OsString::as_os_str);
        }
        if let Some(value) = arg
            .to_str()
            .and_then(|a| a.strip_prefix(name))
            .and_then(|a| a.strip_prefix('='))
        {
            return Some(OsStr::new(value));
        }
    }
    None
}
