use std::fs::File;
use std::io::Read;
use std::time;

use bbgrad::autograd::nn::RNNBuilder;
use bbgrad::autograd::tensor_builder::FloatTensorBuilder;
use ndarray::{ArrayD, IxDyn};
use rand::Rng;

fn char_to_one_hot(c: char, vocab: &[char]) -> ArrayD<f64> {
    let mut arr = vec![0.0; vocab.len()];
    if let Some(idx) = vocab.iter().position(|&v| v == c) {
        arr[idx] = 1.0;
    }
    ArrayD::from_shape_vec(IxDyn(&[vocab.len(), 1]), arr).unwrap()
}

fn sample_from_output(output: &ArrayD<f64>, vocab: &[char]) -> char {
    let probs: Vec<f64> = output.iter().copied().collect();
    let mut rng = rand::rng();
    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return vocab[i];
        }
    }
    vocab[probs.len() - 1]
}

fn main() {
    let vocab = [
        ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
        'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ];
    let vocab_size = vocab.len();

    let input_file = std::env::args().nth(1).unwrap();
    let epoch_count = std::env::args()
        .nth(2)
        .unwrap_or("10".to_string())
        .parse::<i32>()
        .unwrap();
    let learning_rate = std::env::args()
        .nth(3)
        .unwrap_or("0.01".to_string())
        .parse::<f64>()
        .unwrap();
    let batch_size: usize = std::env::args()
        .nth(4)
        .unwrap_or("32".to_string())
        .parse()
        .unwrap();
    let sequence_len = 25;
    let predict_len = 200;

    let mut input_vec: Vec<u8> = Vec::new();
    let mut file_text = File::open(&input_file).expect("could not open input file");
    println!("Vocab: {:?}", vocab);
    println!(
        "Input text len: {}",
        file_text
            .metadata()
            .expect("could not get file metadata")
            .len()
    );

    let rnn = RNNBuilder::new((-1.0, 1.0))
        .with_input_size(vocab_size)
        .with_hidden_size(64)
        .with_output_size(vocab_size)
        .build();

    println!("Pre-training prediction:");
    let mut hidden_state = rnn.init_hidden();
    let mut current_char = 't';
    let mut predicted_text = String::from("t");
    for _ in 0..predict_len {
        let input_tensor = FloatTensorBuilder::new()
            .with_ndarray(char_to_one_hot(current_char, &vocab))
            .with_grad(false)
            .build();
        let (new_hidden_state, output) = rnn.forward_step(input_tensor, hidden_state);
        hidden_state = new_hidden_state;
        current_char = sample_from_output(&output.ndarray(), &vocab);
        predicted_text.push(current_char);
    }
    println!("{}", predicted_text);

    file_text
        .read_to_end(&mut input_vec)
        .expect("could not read input file to vector");

    let text: String = String::from_utf8_lossy(&input_vec)
        .to_lowercase()
        .chars()
        .filter(|c| vocab.contains(c))
        .collect();

    let chars: Vec<char> = text.chars().collect();
    let mut sequences: Vec<&[char]> = Vec::new();
    let mut offset = 0;
    while offset + sequence_len < chars.len() {
        sequences.push(&chars[offset..offset + sequence_len + 1]);
        offset += sequence_len;
    }

    println!(
        "Begin training... ({} sequences, batch size {})",
        sequences.len(),
        batch_size
    );
    let t = time::Instant::now();
    for epoch in 1..=epoch_count {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0.0;

        for batch in sequences.chunks(batch_size) {
            rnn.zero_all_grad();

            let mut batch_loss = 0.0;
            for seq in batch {
                let input_tensors: Vec<_> = seq[..seq.len() - 1]
                    .iter()
                    .map(|&c| {
                        FloatTensorBuilder::new()
                            .with_ndarray(char_to_one_hot(c, &vocab))
                            .with_grad(false)
                            .build()
                    })
                    .collect();

                let expected_tensors: Vec<_> = seq[1..]
                    .iter()
                    .map(|&c| {
                        FloatTensorBuilder::new()
                            .with_ndarray(char_to_one_hot(c, &vocab))
                            .with_grad(false)
                            .build()
                    })
                    .collect();

                batch_loss += rnn.forward_and_loss(input_tensors, expected_tensors);
            }

            rnn.update_params(learning_rate, batch.len() as f64);
            epoch_loss += batch_loss / batch.len() as f64;
            batch_count += 1.0;
        }

        println!(
            "epoch: {}, batches: {}, loss: {}",
            epoch,
            batch_count,
            epoch_loss / batch_count
        );
    }
    println!("Training time: {}s", t.elapsed().as_secs_f64());

    println!("Post-training prediction:");
    let mut hidden_state = rnn.init_hidden();
    let mut current_char = 't';
    let mut predicted_text = String::from("t");
    for _ in 0..predict_len {
        let input_tensor = FloatTensorBuilder::new()
            .with_ndarray(char_to_one_hot(current_char, &vocab))
            .with_grad(false)
            .build();
        let (new_hidden_state, output) = rnn.forward_step(input_tensor, hidden_state);
        hidden_state = new_hidden_state;
        current_char = sample_from_output(&output.ndarray(), &vocab);
        predicted_text.push(current_char);
    }
    println!("{}", predicted_text);
}
