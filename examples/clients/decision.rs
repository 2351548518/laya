use std::{
    env,
    error::Error,
    ffi::OsString,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        return Err("Usage: decision ENDPOINT REQUEST.json".into());
    }
    let mut payload = OsString::from("@");
    payload.push(Path::new(&args[2]).canonicalize()?);
    let mut headers = String::from("Content-Type: application/json\n");
    if let Ok(key) = env::var("LAYA_API_KEY") {
        if key.contains(['\r', '\n']) {
            return Err("LAYA_API_KEY must fit on one line".into());
        }
        headers.push_str(&format!("Authorization: Bearer {key}\n"));
    }
    let mut child = Command::new("curl")
        .args(["--disable", "--fail", "--silent", "--show-error", "--max-time", "120",
               "--proto", "=http,https", "--header", "@-", "--data-binary"])
        .arg(payload)
        .arg("--")
        .arg(&args[1])
        .env_remove("LAYA_API_KEY")
        .stdin(Stdio::piped())
        .spawn()?;
    child.stdin.take().ok_or("Could not pass request headers")?.write_all(headers.as_bytes())?;
    if child.wait()?.success() {
        Ok(())
    } else {
        Err("Decision request failed".into())
    }
}
