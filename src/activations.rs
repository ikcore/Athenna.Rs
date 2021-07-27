
#[derive(Copy, Clone)]
pub enum Activations {
	Sigmoid,
	TanH,
	ReLU,
	LeakyRelu,
	SoftMax,
	Linear
}

pub fn activation_to_string(act:&Activations) -> String {
	match act {
		Activations::Sigmoid => "sigmoid".to_string(),
		Activations::TanH => "tanh".to_string(),
		Activations::ReLU => "relu".to_string(),
		Activations::LeakyRelu => "leakyrelu".to_string(),
		Activations::Linear => "linear".to_string(),
		Activations::SoftMax => "softmax".to_string(),
	}
}

pub fn string_to_activation(val:&str) -> Activations {
	match val {
		"sigmoid" => Activations::Sigmoid,
		"tanh" => Activations::TanH,
		"relu" => Activations::ReLU,
		"leakyrelu" => Activations::LeakyRelu,
		"linear" => Activations::Linear,
		"softmax" => Activations::SoftMax,
		_ => Activations::LeakyRelu
	}
}

pub fn sigmoid(x:f32) -> f32
{
	let k = x.exp();
	k / (1.0 + k)
}

pub fn tanh(x:f32) -> f32
{
	x.tanh()
}

pub fn relu(x:f32) -> f32
{
	match 0.0 >= x {
		true => {0.0},
		false => {x}
	}
}

pub fn leakyrelu(x:f32) -> f32
{
	match 0.0 >= x {
		true => {0.01 * x},
		false => {x}
	}
}

pub fn sigmoid_deriv(x:f32) -> f32
{
	x * (1.0 - x)
}

pub fn tanh_deriv(x:f32) -> f32
{
	1.0 - (x * x)
}

pub fn relu_deriv(x:f32) -> f32
{
	match 0.0 >= x {
		true => {0.0},
		false => {x}
	}
}

pub fn leakyrelu_deriv(x:f32) -> f32
{
	match 0.0 >= x {
		true => {0.01},
		false => {x}
	}
}

pub fn softmax(x:f32) -> f32
{
	x.exp()
}