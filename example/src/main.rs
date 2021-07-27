use athenna::nn::*;
use athenna::activations::*;

fn main() {
  println!("Testing Athenna NN");
  let test_file = &"c:/data/test.athenna".to_string();
	let layers:Vec<usize> = vec!{3,5,3};
	let activations:Vec<Activations> = vec!{ Activations::TanH, Activations::Linear };
  let mut nn = Athenna::new(layers, activations);

  nn.learning_rate = 0.02;

  let x = &vec!{0.2,0.8,0.4};
  let y = &vec!{0.7,0.4,0.5};

	// simulate learning and mutation
	// this model will overfit as there is only one set of data
  for i in 0..1000 {
    if i % 100 == 0 {
      nn.mutate(4, 0.0001);
    }
    nn.back_propagate(x, y);
  }
  let w = nn.feed_forward(x);
  println!("cost: {} | {} {} {}", nn.cost, w[0], w[1], w[2]);
  nn.save(test_file);

  let mut nn = Athenna::load(test_file).unwrap();
  let w = nn.feed_forward(x);
  println!("check reloaded nn matches : {} {} {}", nn.cost, w[0], w[1], w[2]);
}