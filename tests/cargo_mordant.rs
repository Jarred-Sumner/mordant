//! `cargo mordant` end to end: the binaries `cargo test` builds, run the way
//! a user runs them, over a one-crate workspace each test writes afresh.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const CARGO_MORDANT: &str = env!("CARGO_BIN_EXE_cargo-mordant");

const MAIN: &str = "fn fallible() -> Result<u32, u32> {\n    Err(1)\n}\n\n\
                    fn main() {\n    fallible().ok();\n}\n";

fn workspace(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).expect("create the workspace");
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n",
    )
    .expect("write Cargo.toml");
    fs::write(root.join("src/main.rs"), MAIN).expect("write main.rs");
    root
}

fn cargo_mordant(root: &Path) -> Output {
    Command::new(CARGO_MORDANT)
        .arg("mordant")
        .current_dir(root)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env_remove("MORDANT_TOML")
        .env_remove("MORDANT_RUSTFLAGS")
        .output()
        .expect("run cargo-mordant")
}

fn stderr(out: &Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The configuration reaches the lints, and a change to it reruns them on a
/// crate cargo would otherwise have left alone.
#[test]
fn dylint_toml_is_read_and_a_change_to_it_rechecks() {
    let root = workspace("config");
    let first = stderr(&cargo_mordant(&root));
    assert!(first.contains("#[warn(discarded_error)]"), "{first}");

    fs::write(
        root.join("dylint.toml"),
        "[mordant]\ndisabled = [\"discarded_error\"]\n",
    )
    .expect("write dylint.toml");
    let disabled = stderr(&cargo_mordant(&root));
    assert!(disabled.contains("Checking demo"), "{disabled}");
    assert!(!disabled.contains("discarded_error"), "{disabled}");

    fs::remove_file(root.join("dylint.toml")).expect("remove dylint.toml");
    let removed = stderr(&cargo_mordant(&root));
    assert!(removed.contains("#[warn(discarded_error)]"), "{removed}");
}

/// `mordant-action` reads lint names off this, one indented
/// `name  level  description` line per lint under a `mordant` heading.
#[test]
fn list_prints_each_lint_with_its_level() {
    let out = Command::new(CARGO_MORDANT)
        .arg("--list")
        .output()
        .expect("run cargo-mordant --list");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("the list is UTF-8");
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("mordant"));
    let names: Vec<&str> = lines
        .map(|line| {
            assert!(line.starts_with("    "), "{line:?}");
            let mut words = line.split_whitespace();
            let name = words.next().expect("a name");
            assert!(
                matches!(words.next(), Some("warn" | "allow" | "deny" | "forbid")),
                "{line:?}"
            );
            name
        })
        .collect();
    assert!(names.contains(&"discarded_error"), "{names:?}");
    assert!(names.contains(&"unused_pub"), "{names:?}");
}
