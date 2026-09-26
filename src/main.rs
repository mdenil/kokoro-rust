fn main() {
    kokoro::timeline::process_start();
    match kokoro::cli_main() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(2);
        }
    }
}
