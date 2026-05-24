use std::env;
use std::io::{self, Write};
use kona_rs::rag::{Document, RetrievalPipeline, format_hermes_prompt};

fn main() {
    // 1. Define sample documents
    let sample_contents = vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "Project Chimera is a cutting-edge retrieval-augmented generation (RAG) pipeline.",
        "TF-IDF stands for Term Frequency-Inverse Document Frequency, a numerical statistic.",
        "Hermes 2 is a state-of-the-art model that uses ChatML templates for prompt formatting.",
        "Concurrency in Rust is safe and efficient due to ownership and lifetimes.",
    ];

    let mut docs = Vec::new();
    for (i, content) in sample_contents.iter().enumerate() {
        docs.push(Document {
            id: i + 1,
            content: content.to_string(),
        });
    }

    // 2. Initialize the Retrieval Pipeline (fits TF-IDF and builds DB)
    let pipeline = RetrievalPipeline::new(docs);

    // 3. Command Line Arguments Mode
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let query = args[1..].join(" ");
        if !query.trim().is_empty() {
            run_retrieval_and_print(&pipeline, query.trim(), 2);
            return;
        }
    }

    // 4. Quiet Startup & Interactive loop
    println!("Project Chimera RAG CLI. Type 'exit' or 'quit' to exit.");
    loop {
        print!("Query > ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                // EOF
                break;
            }
            Ok(_) => {
                let query = input.trim();
                if query.is_empty() {
                    continue;
                }
                if query.eq_ignore_ascii_case("exit") || query.eq_ignore_ascii_case("quit") {
                    break;
                }

                run_retrieval_and_print(&pipeline, query, 2);
            }
            Err(_) => {
                break;
            }
        }
    }
}

fn run_retrieval_and_print(pipeline: &RetrievalPipeline, query: &str, k: usize) {
    let results = pipeline.retrieve(query, k);
    let mut context_str = String::new();

    for (doc, score) in &results {
        if *score > 0.0 {
            if !context_str.is_empty() {
                context_str.push_str("\n");
            }
            context_str.push_str(&format!("- {}", doc.content));
        }
    }

    let system_prompt = "You are a helpful, respectful, and honest assistant. Answer the user question based on the context provided.";
    let prompt = format_hermes_prompt(system_prompt, &context_str, query);
    println!("{}", prompt);
}
