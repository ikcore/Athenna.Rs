use rand::Rng;
use crate::activations::*;

use std::fs::File;
use std::io::{prelude::*, BufReader, BufWriter};

pub struct Athenna {
	pub neurons:Vec<Vec<f32>>,
	pub weights:Vec<Vec<Vec<f32>>>,
	pub biases:Vec<Vec<f32>>,
	pub activations:Vec<Activations>,
	pub layers:Vec<usize>,
	pub learning_rate: f32,
	pub cost: f32
}

impl Athenna {

	pub fn new(layers:Vec<usize>, activations:Vec<Activations>) -> Athenna
	{
		let mut nn = Athenna {
			neurons: Vec::new(),
			weights: Vec::new(),
			activations,
			layers,
			biases: Vec::new(),
			learning_rate: 0.01,
			cost: 0.0
		};

		nn.init_neurons();
		nn.init_biases();
		nn.init_weights();

		return nn;
	}

	pub fn init_neurons(&mut self)
	{
		self.neurons = Vec::new();
		for x in 0..self.layers.len() {
			let neurons = vec![0.0; self.layers[x]];
			self.neurons.push(neurons);
		}
	}

	pub fn init_biases(&mut self)
	{
		let mut rng = rand::thread_rng();
		self.biases = Vec::with_capacity(self.layers.len());

		for i in 1..self.layers.len() {
			let num_items = self.layers[i];
			let mut bias : Vec<f32> = vec![0.0; num_items];
			for i in 0..num_items {
				bias[i] = rng.gen_range(-0.5..0.5) / num_items as f32;
			}
			self.biases.push(bias);
		}
	}

	pub fn init_weights(&mut self)
	{
		let mut rng = rand::thread_rng();
		self.weights = Vec::new();

		for i in 1..self.layers.len() {
			let num_prev_items = self.layers[i-1];
			let mut layer_weights : Vec<Vec<f32>> = Vec::new();
			for _j in 0..self.layers[i] {
				let mut neuron_weights : Vec<f32> = vec![0.0; num_prev_items];
				for k in 0..num_prev_items {
					neuron_weights[k] = rng.gen_range(-0.5..0.5) / num_prev_items as f32;
				}
				layer_weights.push(neuron_weights);
			}
			self.weights.push(layer_weights);
		}
	}

	pub fn activate(&self, x:f32, layer_id: usize) -> f32 {
		match self.activations[layer_id] {
			Activations::Sigmoid => sigmoid(x),
			Activations::ReLU => relu(x),
			Activations::LeakyRelu => leakyrelu(x),
			Activations::TanH => tanh(x),
			Activations::SoftMax => softmax(x),
			_ => x
		}
	}

	pub fn activate_deriv(&self, x:f32, layer_id: usize) -> f32 {
		match self.activations[layer_id] {
			Activations::Sigmoid => sigmoid_deriv(x),
			Activations::ReLU => relu_deriv(x),
			Activations::LeakyRelu => leakyrelu_deriv(x),
			Activations::TanH => tanh_deriv(x),
			Activations::SoftMax => softmax(x),
			_ => x
		}
	}

	pub fn feed_forward(&mut self, inputs:&Vec<f32>) -> Vec<f32> {
		for i in 0..inputs.len() {
			self.neurons[0][i] = inputs[i];
		}
		for i in 1..inputs.len() {
			let layer_idx = i - 1;

			for j in 0..self.layers[i] {
				let mut value:f32 = 0.0;
				for k in 0..self.layers[i-1] {
					value += self.weights[i - 1][j][k] * self.neurons[i - 1][k];
				}
				self.neurons[i][j] = self.activate(value + self.biases[i - 1][j], layer_idx);
			}

			match self.activations[layer_idx] {
				Activations::SoftMax => {
					let mut sigma : f32 = 0.0;
					for j in 0..self.layers[i] {
						sigma += self.neurons[i][j];
					}
					for j in 0..self.layers[i] {
						self.neurons[i][j] /= sigma;
					}
				},
				_ => {}
			}
		}
		self.neurons[self.layers.len() - 1].clone()
	}

	pub fn back_propagate(&mut self, inputs:&Vec<f32>, expected:&Vec<f32>) {	
		let output = self.feed_forward(inputs);
		let mut cost : f32 = 0.0;

		for i in 0..output.len() {
			cost += f32::powf(output[i] - expected[i], 2.0);
		}
		self.cost = cost / 2.0;

		let mut gamma : Vec<Vec<f32>> = Vec::new();
		for i in 0..self.layers.len() {
			gamma.push(vec![0.0; self.layers[i]]);
		}
		let layer = self.layers.len() - 2;
		let last_layer_idx = self.layers.len() - 1;

		match self.activations[layer] {
			Activations::SoftMax => {
				for i in 0..output.len() {
					gamma[last_layer_idx][i] = (output[i] - expected[i]) * (output[i] * (1.0 - output[i]));
				}
			},
			Activations::Linear => {
				for i in 0..output.len() {
					gamma[last_layer_idx][i] = output[i] - expected[i];
				}
			},
			_ => {
				for i in 0..output.len() {
					gamma[last_layer_idx][i] = (output[i] - expected[i]) * self.activate(output[i], layer);
				}	
			}
		}

		for i in 0..self.layers[last_layer_idx] {
			self.biases[self.layers.len()-2][i]
				= gamma[self.layers.len()-1][i] * self.learning_rate;

			for j in 0..(self.layers.len() - 2) {
				self.weights[self.layers.len()-2][i][j]
					-= gamma[self.layers.len()-1][i]
						* self.neurons[self.layers.len()-2][j]
						* self.learning_rate
			}
		}
		for i in (1..self.layers.len()-1).rev() {
			let layer = i - 1;
			for j in 0..self.layers[i] {
				gamma[i][j] = 0.0;
				for k in 0..gamma[i+1].len() {
					gamma[i][j] += gamma[i + 1][k] * self.weights[i][k][j];
				}
				gamma[i][j] *= self.activate_deriv(self.neurons[i][j], layer);
			}
			for j in 0..self.layers[i] { 
				self.biases[i - 1][j] -= gamma[i][j] * self.learning_rate;
				for k in 0..(self.layers.len()-1) {
					self.weights[i - 1][j][k] -= gamma[i][j] * self.neurons[i - 1][k] * self.learning_rate;
				}
			}
		}
	}
	pub fn mutate(&mut self, high:i32, val:f32)
	{
		let mut rng = rand::thread_rng();
		for i in 0..self.biases.len() {
			for j in 0..self.biases[i].len() {
				if rng.gen_range(0.0..(high as f32)) <= 2.0 {
					self.biases[i][j] += rng.gen_range(-val..val);
				}
			}
		}
		for i in 0..self.weights.len() {
			for j in 0..self.weights[i].len() {
				for k in 0..self.weights[i][j].len() {
					if rng.gen_range(0.0..(high as f32)) <= 2.0 {
						self.weights[i][j][k] += rng.gen_range(-val..val);
					}
				}
			}
		}
	}

	pub fn save(&self, path:&String) {

		let file = File::create(path).unwrap();
		let mut writer = BufWriter::new(file);

		let _ = writer.write("[network.type]\nathenna.v2".as_bytes());
		let _ = writer.write(format!("\n\n[dims]\n{}\n{}",
			self.layers[0].to_string(),
			self.layers[self.layers.len()-1].to_string()
		).as_bytes());
		let _ = writer.write(format!("\n\n[layer.total]\n{}",self.layers.len().to_string()).as_bytes());

		let _ = writer.write("\n\n[layer.neurons]".as_bytes());
		for i in 0..self.layers.len()  {
			let _ = writer.write(format!("\n{}", self.layers[i].to_string() ).as_bytes());
		}

		let _ = writer.write("\n\n[layer.activations]".as_bytes());
		for i in 0..self.layers.len() - 1 {
			let _ = writer.write(format!("\n{}", &activation_to_string(&self.activations[i])).as_bytes());
		}
		let _ = writer.write("\n\n[layer.weights]".as_bytes());

		for i in 0..self.weights.len() {
			for j in 0..self.weights[i].len() {
				for k in 0..self.weights[i][j].len() {
					let _ = writer.write(format!("\n{}", self.weights[i][j][k].to_string()).as_bytes());
				}
			}
		}
		let _ = writer.write("\n\n[layer.biases]".as_bytes());
		for i in 0..self.biases.len() {
			for j in 0..self.biases[i].len() {
				let _ = writer.write(format!("\n{}", self.biases[i][j].to_string()).as_bytes());
			}
		}

		let _ = writer.write("\n\n[network.learning_rate]".as_bytes());
		let _ = writer.write(format!("\n{}", self.learning_rate.to_string()).as_bytes());
		let _ = writer.flush();
	}

	pub fn load(path:&String) -> Result<Athenna, std::io::Error> {

		let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

		let mut nn = Athenna {
			neurons: Vec::new(),
			weights: Vec::new(),
			activations: vec!{Activations::SoftMax},
			layers: vec!{3,4},
			biases: Vec::new(),
			learning_rate: 0.01,
			cost: 0.0
		};

		let mut layers : Vec<usize> = Vec::new();
		let mut activations : Vec<Activations> = Vec::new();
		let mut weights : Vec<f32> = Vec::new();
		let mut biases : Vec<f32> = Vec::new();

		let mut label_idx = 0;
		let mut bias_idx = 0;
		let mut weight_idx = 0;

    for line in reader.lines() {

			let parsed = line.unwrap();
			let content = parsed.as_str();
 
			match content {
				"[network.type]" => {label_idx = 1},
				"[layer.neurons]" => {label_idx = 2},
				"[layer.activations]" => {label_idx = 3},
				"[layer.weights]" => {label_idx = 4},
				"[layer.biases]" => {label_idx = 5},
				"[network.learning_rate]" => {label_idx = 6},
				"" => {
					if (label_idx == 2 || label_idx == 3)
						&& activations.len() == layers.len() - 1
						&& activations.len() > 0 {
							nn = Athenna::new(layers.clone(), activations.clone());
							label_idx = 0;
						}
				},
				_ => {
					match label_idx {
						2 => {
							if content != "" {
								layers.push(content.parse::<usize>().unwrap());
							}
						},
						3 => {
							if content != "" {
								activations.push(string_to_activation(content));
							}	
						},
						4 => {
							if content != "" {
								weights.push(content.parse::<f32>().unwrap());
							}
						},
						5 => {
							if content != "" {
								biases.push(content.parse::<f32>().unwrap());
							}
						},
						6 => {
							if content != "" {
								nn.learning_rate = content.parse::<f32>().unwrap();
							}
						}
						_ => {}
					}
				}
			}
		}
		for i in 0..nn.biases.len() {
			for j in 0..nn.biases[i].len() {
				nn.biases[i][j] = biases[bias_idx];
				bias_idx += 1;
			}
		}
		for i in 0..nn.weights.len() {
			for j in 0..nn.weights[i].len() {
				for k in 0..nn.weights[i][j].len() {
					nn.weights[i][j][k] = weights[weight_idx];
					weight_idx += 1;
				}
			}
		}
		return Ok(nn)
	}
}
