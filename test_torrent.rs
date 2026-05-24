use std::env;
use std::io::{self, Write};
use std::thread;
use std::time::{Duration, Instant};

use kona_rs::k::K;
use kona_rs::va;
use kona_rs::parallel;

struct SimpleRng {
    state: u32,
}

impl SimpleRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    fn next_bool(&mut self, probability: f64) -> bool {
        let val = (self.next_u32() & 0xFFFF) as f64 / 65536.0;
        val < probability
    }
}

struct Seeder {
    id: usize,
    latency_ms: u64,
    available_pieces: K,
}

fn init_swarm(num_pieces: usize, num_seeders: usize) -> Vec<Seeder> {
    let mut rng = SimpleRng::new(42); // fixed seed for deterministic behavior
    let mut seeders = Vec::new();
    for id in 1..=num_seeders {
        let latency_ms = (id as u64 * 15) % 80 + 10; // deterministic latencies
        let mut bits = Vec::with_capacity(num_pieces);
        for _ in 0..num_pieces {
            // Seeders have a 60% chance of having any given piece
            bits.push(if rng.next_bool(0.6) { 1 } else { 0 });
        }
        seeders.push(Seeder {
            id,
            latency_ms,
            available_pieces: K::from_ints(bits),
        });
    }
    seeders
}

fn run_broadcast_gather(seeders: &[Seeder]) -> K {
    if seeders.is_empty() {
        return K::from_ints(Vec::new());
    }

    // Spawn threads in parallel to query each seeder
    let gathered = thread::scope(|s| {
        let mut handles = Vec::new();
        for seeder in seeders {
            handles.push(s.spawn(move || {
                // Simulate latency
                if seeder.latency_ms > 0 {
                    thread::sleep(Duration::from_millis(seeder.latency_ms));
                }
                seeder.available_pieces.clone()
            }));
        }
        let mut results = Vec::new();
        for h in handles {
            results.push(h.join().unwrap());
        }
        results
    });

    // Gather by summing up the K objects using va::plus
    let mut gathered_sum = gathered[0].clone();
    for k_obj in &gathered[1..] {
        gathered_sum = va::plus(&gathered_sum, k_obj);
    }

    gathered_sum
}

fn print_swarm_status(seeders: &[Seeder]) {
    if seeders.is_empty() {
        println!("No seeders in swarm.");
        return;
    }
    println!("Swarm Status ({} Seeders, {} Pieces):", seeders.len(), seeders[0].available_pieces.n);
    for seeder in seeders {
        let data = seeder.available_pieces.ki_data();
        let latency = seeder.latency_ms;
        if data.len() <= 20 {
            println!("  Seeder #{:<2} (latency: {:>2}ms): {:?}", seeder.id, latency, data);
        } else {
            println!(
                "  Seeder #{:<2} (latency: {:>2}ms): [{:?}, ..., {:?}]",
                seeder.id, latency, &data[..5], &data[data.len()-5..]
            );
        }
    }
}

fn select_pieces(
    strategy: &str,
    client_pieces: &[bool],
    availability: &[i64],
    priorities: &[i64],
) -> Vec<usize> {
    use kona_rs::k::K;
    let own_ints: Vec<i64> = client_pieces.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let own_bitfield = K::from_ints(own_ints);
    let peer_bitfields = K::from_ints(availability.to_vec());
    let policy_id = match strategy.to_lowercase().as_str() {
        "sequential" | "seq" => 1,
        "rarest" | "rarest-first" => 0,
        "priority" | "priority-weighted" => 2,
        _ => 1,
    };
    let policy = K::ki(policy_id);
    let weights_floats: Vec<f64> = priorities.iter().map(|&x| x as f64).collect();
    let custom_weights = K::from_floats(weights_floats);
    let res = kona_rs::torrent::select_pieces(&peer_bitfields, &own_bitfield, &policy, Some(&custom_weights));
    res.ki_data().iter().map(|&x| x as usize).collect()
}

fn run_demo() {
    let num_pieces = 10;
    let num_seeders = 5;
    println!("=== BitTorrent Swarm Demo ===");
    let seeders = init_swarm(num_pieces, num_seeders);
    print_swarm_status(&seeders);

    println!("\nRunning broadcast-gather query (in parallel)...");
    let gathered_sum = run_broadcast_gather(&seeders);
    let avail = gathered_sum.ki_data();
    println!("Gathered Swarm Availability: {:?}", avail);

    let mut client_pieces = vec![false; num_pieces];
    client_pieces[2] = true;
    client_pieces[5] = true;
    println!("\nClient Piece Status (1=Have, 0=Missing):");
    println!("  {:?}", client_pieces.iter().map(|&b| if b { 1 } else { 0 }).collect::<Vec<i64>>());

    let mut priorities = vec![2; num_pieces];
    priorities[0] = 3; // High priority
    priorities[9] = 1; // Low priority
    println!("Piece Priorities (1=Low, 2=Medium, 3=High):");
    println!("  {:?}", priorities);

    // Test strategies
    for strategy in &["sequential", "rarest", "priority"] {
        let order = select_pieces(strategy, &client_pieces, avail, &priorities);
        println!("  Strategy: {:<12} -> Download order: {:?}", strategy, order);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Check for demo or bench commands via command line arguments
    if args.len() > 1 {
        let arg = args[1].trim().to_lowercase();
        if arg == "demo" {
            run_demo();
            return;
        } else if arg == "bench" {
            let num_pieces = 200_000;
            let num_seeders = 8;
            println!("Initializing large benchmark: {} pieces, {} seeders...", num_pieces, num_seeders);
            let seeders = init_swarm(num_pieces, num_seeders);
            println!("Running parallel broadcast-gather query...");
            let start = Instant::now();
            let gathered_sum = run_broadcast_gather(&seeders);
            let duration = start.elapsed();
            println!("Gather completed in {:.2?} using {} workers.", duration, parallel::worker_count(num_pieces));
            println!("Verification of parallel gather result (first 5 elements): {:?}", &gathered_sum.ki_data()[..5]);
            return;
        } else {
            println!("Unknown option '{}'. Use 'demo' to run verification, 'bench' to run benchmark, or no arguments for interactive mode.", args[1]);
            return;
        }
    }

    // Interactive Loop
    let mut num_pieces = 10;
    let mut num_seeders = 5;
    let mut seeders = init_swarm(num_pieces, num_seeders);
    let mut client_pieces = vec![false; num_pieces];
    if num_pieces > 5 {
        client_pieces[2] = true;
        client_pieces[5] = true;
    }
    let mut priorities = vec![2; num_pieces];
    let mut availability = vec![0; num_pieces];
    let mut query_run = false;

    println!("BitTorrent Swarm Validation CLI. Type 'help' for commands.");
    loop {
        print!("Swarm > ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break; // EOF
        }
        let input_trimmed = input.trim();
        if input_trimmed.is_empty() {
            if !input.is_empty() {
                // Sent newline only, keep prompt going
                continue;
            }
            break; // EOF
        }
        let parts: Vec<&str> = input_trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let cmd = parts[0].to_lowercase();
        match cmd.as_str() {
            "exit" | "quit" => {
                println!("Exiting Swarm CLI. Goodbye!");
                break;
            }
            "help" => {
                println!("Commands:");
                println!("  status                      - Show current client pieces, priorities, and swarm state");
                println!("  swarm                       - Show seeders and their piece availability");
                println!("  query                       - Run parallel query broadcast-gather");
                println!("  select <strategy>           - Run piece selection (rarest, seq, priority)");
                println!("  set_pieces <N>              - Re-initialize swarm with N pieces");
                println!("  set_seeders <N>             - Re-initialize swarm with N seeders");
                println!("  set_priority <idx> <prio>   - Set piece priority (1=Low, 2=Medium, 3=High)");
                println!("  set_client <idx> <0|1>      - Set client ownership of a piece");
                println!("  bench                       - Run high-performance parallel scheduler benchmark");
                println!("  exit / quit                 - Exit the CLI");
            }
            "status" => {
                println!("Client Pieces (1=Have, 0=Missing):");
                println!("  {:?}", client_pieces.iter().map(|&b| if b { 1 } else { 0 }).collect::<Vec<i64>>());
                println!("Piece Priorities (1=Low, 2=Medium, 3=High):");
                println!("  {:?}", priorities);
                if query_run {
                    println!("Last Gathered Swarm Availability:");
                    println!("  {:?}", availability);
                } else {
                    println!("Swarm availability has not been queried yet. Run 'query' first.");
                }
            }
            "swarm" => {
                print_swarm_status(&seeders);
            }
            "query" => {
                println!("Broadcasting to {} seeders in parallel...", seeders.len());
                let start = Instant::now();
                let gathered_sum = run_broadcast_gather(&seeders);
                let duration = start.elapsed();
                availability = gathered_sum.ki_data().to_vec();
                query_run = true;
                println!("Gathered Swarm Availability: {:?}", availability);
                println!("Query broadcast-gather took {:.2?}", duration);
            }
            "select" => {
                if parts.len() < 2 {
                    println!("Error: specify strategy (rarest, seq, priority)");
                    continue;
                }
                if !query_run {
                    println!("Running query first to get piece availability...");
                    let gathered_sum = run_broadcast_gather(&seeders);
                    availability = gathered_sum.ki_data().to_vec();
                    query_run = true;
                }
                let strategy = parts[1];
                let order = select_pieces(strategy, &client_pieces, &availability, &priorities);
                println!("Strategy: {} -> Download order: {:?}", strategy, order);
            }
            "set_pieces" => {
                if parts.len() < 2 {
                    println!("Error: specify number of pieces");
                    continue;
                }
                if let Ok(n) = parts[1].parse::<usize>() {
                    if n == 0 {
                        println!("Error: pieces must be > 0");
                        continue;
                    }
                    num_pieces = n;
                    seeders = init_swarm(num_pieces, num_seeders);
                    client_pieces = vec![false; num_pieces];
                    priorities = vec![2; num_pieces];
                    availability = vec![0; num_pieces];
                    query_run = false;
                    println!("Re-initialized swarm with {} pieces.", num_pieces);
                } else {
                    println!("Error: invalid number");
                }
            }
            "set_seeders" => {
                if parts.len() < 2 {
                    println!("Error: specify number of seeders");
                    continue;
                }
                if let Ok(n) = parts[1].parse::<usize>() {
                    if n == 0 {
                        println!("Error: seeders must be > 0");
                        continue;
                    }
                    num_seeders = n;
                    seeders = init_swarm(num_pieces, num_seeders);
                    query_run = false;
                    println!("Re-initialized swarm with {} seeders.", num_seeders);
                } else {
                    println!("Error: invalid number");
                }
            }
            "set_priority" => {
                if parts.len() < 3 {
                    println!("Error: specify <idx> and <priority>");
                    continue;
                }
                let idx = parts[1].parse::<usize>();
                let prio = parts[2].parse::<i64>();
                match (idx, prio) {
                    (Ok(i), Ok(p)) if i < num_pieces && (1..=3).contains(&p) => {
                        priorities[i] = p;
                        println!("Set piece {} priority to {}.", i, p);
                    }
                    _ => println!("Error: invalid index (0-{}) or priority (1-3)", num_pieces - 1),
                }
            }
            "set_client" => {
                if parts.len() < 3 {
                    println!("Error: specify <idx> and <0 or 1>");
                    continue;
                }
                let idx = parts[1].parse::<usize>();
                let val = parts[2].parse::<i64>();
                match (idx, val) {
                    (Ok(i), Ok(v)) if i < num_pieces && (v == 0 || v == 1) => {
                        client_pieces[i] = v == 1;
                        println!("Set client piece {} ownership to {}.", i, v);
                    }
                    _ => println!("Error: invalid index (0-{}) or value (0 or 1)", num_pieces - 1),
                }
            }
            "bench" => {
                let bench_pieces = 200_000;
                let bench_seeders = 8;
                println!("Initializing large benchmark: {} pieces, {} seeders...", bench_pieces, bench_seeders);
                let bench_swarm = init_swarm(bench_pieces, bench_seeders);
                println!("Running parallel broadcast-gather query...");
                let start = Instant::now();
                let gathered_sum = run_broadcast_gather(&bench_swarm);
                let duration = start.elapsed();
                println!("Gather completed in {:.2?} using {} workers.", duration, parallel::worker_count(bench_pieces));
                println!("Verification of parallel gather result (first 5 elements): {:?}", &gathered_sum.ki_data()[..5]);
            }
            _ => {
                println!("Unknown command: '{}'. Type 'help' for commands.", cmd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential() {
        let client_pieces = vec![false, false, true, false, false];
        let availability = vec![2, 3, 4, 1, 5];
        let priorities = vec![2, 2, 2, 2, 2];
        let order = select_pieces("sequential", &client_pieces, &availability, &priorities);
        assert_eq!(order, vec![0, 1, 3, 4]);
    }

    #[test]
    fn test_rarest_first() {
        let client_pieces = vec![false, false, true, false, false];
        let availability = vec![2, 3, 4, 1, 5];
        let priorities = vec![2, 2, 2, 2, 2];
        let order = select_pieces("rarest", &client_pieces, &availability, &priorities);
        // rarest order should be 3 (avail 1), 0 (avail 2), 1 (avail 3), 4 (avail 5)
        assert_eq!(order, vec![3, 0, 1, 4]);
    }

    #[test]
    fn test_rarest_first_differing_availability() {
        let client_pieces = vec![false, false, false, false];
        let availability = vec![5, 0, 2, 1]; // piece 1 is 0 availability (undownloadable)
        let priorities = vec![2, 2, 2, 2];
        let order = select_pieces("rarest", &client_pieces, &availability, &priorities);
        // piece 1 has 0 availability so it should be excluded
        assert_eq!(order, vec![3, 2, 0]);
    }

    #[test]
    fn test_priority_weighted() {
        let client_pieces = vec![false, false, false, false];
        let availability = vec![2, 2, 4, 1];
        let priorities = vec![2, 3, 2, 2]; // piece 1 has High priority (3)
        let order = select_pieces("priority", &client_pieces, &availability, &priorities);
        // piece 1 (priority 3, avail 2) -> first
        // then priority 2 pieces:
        //   piece 3 (avail 1) -> second
        //   piece 0 (avail 2) -> third
        //   piece 2 (avail 4) -> fourth
        assert_eq!(order, vec![1, 3, 0, 2]);
    }
}
